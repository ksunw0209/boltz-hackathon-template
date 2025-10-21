import gc
from typing import Any, Optional
import math

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.mol import (
    minimum_lddt_symmetry_coords,
)
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.loss.bfactor import bfactor_loss_fn
from boltz.model.loss.confidencev2 import (
    confidence_loss,
)
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.modules.affinity import AffinityModule
from boltz.model.modules.confidencev2 import ConfidenceModule
from boltz.model.modules.confidence import ConfidenceModule as ConfidenceModuleV1
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
from boltz.model.modules.diffusionv2 import AtomDiffusion
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.trunkv2 import (
    BFactorModule,
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
    TemplateModule,
    TemplateV2Module,
)
from boltz.model.optim.ema import EMA
from boltz.model.optim.scheduler import AlphaFoldLRScheduler
from boltz.model.models.boltz1 import Boltz1


class Boltz2(LightningModule):
    """Boltz2 model."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args1: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        validators: Any = None,
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[dict] = None,
        confidence_prediction: bool = True,
        affinity_prediction: bool = False,
        affinity_ensemble: bool = False,
        affinity_mw_correction: bool = True,
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = False,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        validate_structure: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        compile_affinity: bool = False,
        compile_msa: bool = False,
        exclude_ions_from_lddt: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        use_no_atom_char: bool = False,
        no_random_recycling_training: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        steering_args: Optional[dict] = None,
        logmd_args: Optional[dict] = None,
        use_templates: bool = False,
        compile_templates: bool = False,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        use_templates_v2: bool = False,
        use_kernels: bool = False,
        use_dropout: bool = False,
        boltz1_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["validators"])

        # Flag to disable state changes when using DataParallel
        self.is_in_dataparallel = False

        # No random recycling
        self.no_random_recycling_training = no_random_recycling_training

        if validate_structure:
            # Late init at setup time
            self.val_group_mapper = {}  # maps a dataset index to a validation group name
            self.validator_mapper = {}  # maps a dataset index to a validator

            # Validators for each dataset keep track of all metrics,
            # compute validation, aggregate results and log
            self.validators = nn.ModuleList(validators)

        self.num_val_datasets = num_val_datasets
        self.log_loss_every_steps = log_loss_every_steps

        # EMA
        self.use_ema = ema
        self.ema_decay = ema_decay

        # Arguments
        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args
        self.steering_args = steering_args
        self.logmd_args = logmd_args
        # Training metrics
        if validate_structure:
            self.train_confidence_loss_logger = MeanMetric()
            self.train_confidence_loss_dict_logger = nn.ModuleDict()
            for m in [
                "plddt_loss",
                "resolved_loss",
                "pde_loss",
                "pae_loss",
            ]:
                self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.exclude_ions_from_lddt = exclude_ions_from_lddt

        # Distogram
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.aggregate_distogram = aggregate_distogram

        # Trunk
        self.is_pairformer_compiled = False
        self.is_msa_compiled = False
        self.is_template_compiled = False

        # Trifast
        self.use_kernels = use_kernels
        self.use_dropout = use_dropout

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
        )

        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.bond_type_feature = bond_type_feature
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Set compile rules
        # Big models hit the default cache limit (8)
        torch._dynamo.config.cache_size_limit = 512  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001

        # Pairwise stack
        self.use_templates = use_templates
        if use_templates:
            if use_templates_v2:
                self.template_module = TemplateV2Module(token_z, **template_args)
            else:
                self.template_module = TemplateModule(token_z, **template_args)
            if compile_templates:
                self.is_template_compiled = True
                self.template_module = torch.compile(
                    self.template_module,
                    dynamic=False,
                    fullgraph=False,
                )

        self.msa_module = MSAModule(
            token_z=token_z,
            token_s=token_s,
            **msa_args,
        )
        if compile_msa:
            self.is_msa_compiled = True
            self.msa_module = torch.compile(
                self.msa_module,
                dynamic=False,
                fullgraph=False,
            )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            self.is_pairformer_compiled = True
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=score_model_args["atom_encoder_depth"],
            atom_encoder_heads=score_model_args["atom_encoder_heads"],
            token_transformer_depth=score_model_args["token_transformer_depth"],
            token_transformer_heads=score_model_args["token_transformer_heads"],
            atom_decoder_depth=score_model_args["atom_decoder_depth"],
            atom_decoder_heads=score_model_args["atom_decoder_heads"],
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=score_model_args[
                "conditioning_transition_layers"
            ],
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )


        self.confidence_prediction = confidence_prediction
        self.affinity_prediction = affinity_prediction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction
        self.run_trunk_and_structure = run_trunk_and_structure
        self.skip_run_structure = skip_run_structure
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae
        self.structure_prediction_training = structure_prediction_training

        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(
                token_s,
                token_z,
                token_level_confidence=token_level_confidence,
                bond_type_feature=bond_type_feature,
                fix_sym_check=fix_sym_check,
                cyclic_pos_enc=cyclic_pos_enc,
                conditioning_cutoff_min=conditioning_cutoff_min,
                conditioning_cutoff_max=conditioning_cutoff_max,
                **confidence_model_args,
            )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )
        
        if self.steering_args and self.steering_args.get("use_boltz1_confidence_steering", False):
            assert (
                boltz1_checkpoint is not None
            ), "boltz1_checkpoint must be provided for confidence steering"
            print("Loading boltz1 model for confidence steering")
            # Restore Boltz1 directly from checkpoint to avoid constructor arg mismatches            
            self.boltz1_model_container = [
                Boltz1.load_from_checkpoint(
                    boltz1_checkpoint,
                    map_location="cpu",
                    strict=True,
                ).eval()
            ]
            for param in self.boltz1_model_container[0].parameters():
                param.requires_grad = False

            # Optionally keep trunk for features; otherwise prune heavy modules to save memory
            if not self.steering_args.get("use_boltz1_trunk_features", False):
                boltz1_model = self.boltz1_model_container[0]
                for attr_name in [
                    "input_embedder",
                    "rel_pos",
                    "token_bonds",
                    "s_init",
                    "z_init_1",
                    "z_init_2",
                    "s_norm",
                    "z_norm",
                    "s_recycle",
                    "z_recycle",
                    "msa_module",
                    "pairformer_module",
                    "structure_module",
                    "distogram_module",
                ]:
                    if hasattr(boltz1_model, attr_name):
                        delattr(boltz1_model, attr_name)
                # Encourage freeing unused memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print("Finished loading boltz1 model for confidence steering")

        # Output modules
        use_accumulate_token_repr = False # (
        #     self.steering_args.get("use_boltz1_confidence_steering", False)
        #     and self.boltz1_model_container[0].confidence_module.use_s_diffusion
        # )
        diffusion_process_args.pop("mse_rotational_alignment", None)
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                **score_model_args,
            },
            compile_score=compile_structure,
            accumulate_token_repr=use_accumulate_token_repr,
            **diffusion_process_args,
        )
        self.distogram_module = DistogramModule(
            token_z,
            num_bins,
        )
        self.predict_bfactor = predict_bfactor
        if predict_bfactor:
            self.bfactor_module = BFactorModule(token_s, num_bins)

        if self.affinity_prediction:
            if self.affinity_ensemble:
                self.affinity_module1 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args1,
                )
                self.affinity_module2 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args2,
                )
                if compile_affinity:
                    self.affinity_module1 = torch.compile(
                        self.affinity_module1, dynamic=False, fullgraph=False
                    )
                    self.affinity_module2 = torch.compile(
                        self.affinity_module2, dynamic=False, fullgraph=False
                    )
            else:
                self.affinity_module = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args,
                )
                if compile_affinity:
                    self.affinity_module = torch.compile(
                        self.affinity_module, dynamic=False, fullgraph=False
                    )

        # Remove grad from weights they are not trained for ddp
        if not structure_prediction_training:
            for name, param in self.named_parameters():
                if (
                    name.split(".")[0] not in ["confidence_module", "affinity_module"]
                    and "out_token_feat_update" not in name
                ):
                    param.requires_grad = False

    def setup(self, stage: str) -> None:
        """Set the model for training, validation."""
        if (
            stage != "predict"
            and hasattr(self.trainer, "datamodule")
            and self.trainer.datamodule
            and self.validate_structure
        ):
            self.val_group_mapper.update(self.trainer.datamodule.val_group_mapper)

            l1 = len(self.val_group_mapper)
            l2 = self.num_val_datasets
            msg = (
                f"Number of validation datasets num_val_datasets={l2} "
                f"does not match the number of val_group_mapper entries={l1}."
            )
            assert l1 == l2, msg

            # Map an index to a validator, and double check val names
            # match from datamodule
            all_validator_names = []
            for validator in self.validators:
                for val_name in validator.val_names:
                    msg = f"Validator {val_name} duplicated in validators."
                    assert val_name not in all_validator_names, msg
                    all_validator_names.append(val_name)
                    for val_idx, val_group in self.val_group_mapper.items():
                        if val_name == val_group["label"]:
                            self.validator_mapper[val_idx] = validator

            msg = "Mismatch between validator names and val_group_mapper values."
            assert set(all_validator_names) == {
                x["label"] for x in self.val_group_mapper.values()
            }, msg

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        s_z_samples: int = 1,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        max_multiplicity: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        print(feats["record"][0].id)

        s_list = []
        z_list = []
        for i in range(s_z_samples):
            print("s_z_sample", i)
            s_inputs, s, z, relative_position_encoding = self.trunk_forward(
                feats=feats,
                recycling_steps=recycling_steps,
            )
            s_list.append(s.detach().clone().float())
            z_list.append(z.detach().clone().float())

        # Offload pairformer module to CPU to save GPU memory
        if not self.training and s_z_samples > 1:
            self.pairformer_module = self.pairformer_module.cpu()
            torch.cuda.empty_cache()

        pdistogram_list = []
        for z in z_list:
            pdistogram_list.append(self.distogram_module(z))
        pdistogram = torch.cat(pdistogram_list, dim=3) # Treat different samples as different conformers
        dict_out = {"pdistogram": pdistogram}

        if (
            (self.predict_args.get("structure_ablation", False) or self.predict_args.get("confidence_ablation", False))
            and not self.training
        ):
            assert s_z_samples == 1, "Structure ablation only supports 1 sample"

            # Predicted distogram
            pred_distogram_logits = dict_out["pdistogram"][:, :, :, 0].detach()
            pred_logits_min = pred_distogram_logits.min()
            pred_logits_max = pred_distogram_logits.max()
            print("pred_distogram_logits", torch.sum(torch.argmax(pred_distogram_logits, dim=-1)))
            print("pred_distogram_logits", pred_distogram_logits)
            print("pred_distogram_logits mean", pred_distogram_logits.mean())

            # Aggregate distogram over K conformers
            assert len(feats["disto_target"].shape) == 5
            print(feats["disto_target"].shape)
            disto_target = feats["disto_target"].sum(dim=3).detach()  # (1, L, L, n_bins)
            
            '''
            disto_target_logits = torch.where(disto_target == 1., pred_logits_max, pred_logits_min)
            disto_target = torch.argmax(disto_target_logits, dim=-1)
            gt_z = self.confidence_module.dist_bin_pairwise_embed(disto_target)

            '''

            # Get peak indices and adjacent bins
            # peak_indices = torch.argmax(pred_distogram_logits, dim=-1)  # B x L x L
            # num_bins = pred_distogram_logits.shape[-1]
        
            # # Get indices of adjacent bins (-2, -1, 0, +1, +2 relative to peak)
            # indices_offset = torch.tensor([-2, -1, 0, 1, 2], device=peak_indices.device)
            # peak_and_adjacent = peak_indices[..., None] + indices_offset[None, None, None, :]
            # peak_and_adjacent = torch.clamp(peak_and_adjacent, min=0, max=num_bins-1)
        
            # # Get the logit values at these indices
            # B, L1, L2 = pred_distogram_logits.shape[:3]
            # batch_indices = torch.arange(B).view(-1, 1, 1, 1).expand(-1, L1, L2, 5)
            # row_indices = torch.arange(L1).view(1, -1, 1, 1).expand(B, -1, L2, 5)
            # col_indices = torch.arange(L2).view(1, 1, -1, 1).expand(B, L1, -1, 5)
        
            # peak_logit_values = pred_distogram_logits[batch_indices, row_indices, col_indices, peak_and_adjacent]
            # print("Logit values at [peak-2, peak-1, peak, peak+1, peak+2] bins:", peak_logit_values)
            # pred_distogram_delta = (peak_logit_values.sort(dim=-1)[0][..., 1:] - peak_logit_values.sort(dim=-1)[0][..., :-1]).mean()
            # print("pred_distogram_delta", pred_distogram_delta)

            # Ground truth distogram
            # Aggregate distogram over K conformers
            assert len(feats["disto_target"].shape) == 5
            print(feats["disto_target"].shape)
            disto_target = feats["disto_target"].sum(dim=3).detach()  # (1, L, L, n_bins)
            disto_target = disto_target.repeat(diffusion_samples, 1, 1, 1)

            pred_logits_mean = pred_distogram_logits.mean()
            pred_logits_min = pred_distogram_logits.min()
            pred_logits_max = pred_distogram_logits.max()
            distogram_scale = pred_logits_max - pred_logits_min

            print("pred_logits_mean", pred_logits_mean)
            print("pred_logits_min", pred_logits_min)
            print("pred_logits_max", pred_logits_max)
            print("distogram_scale", distogram_scale)

            ############################ Smooth Distogram ############################

            # Convert gt probabilities to logits, using the predicted range
            disto_target_logits = torch.where(disto_target == 1., pred_logits_max, pred_logits_min)
            print("disto_target_logits", disto_target_logits)

            smoothed_disto_target_logits = disto_target_logits
            kernel_size = 5  # Fixed size for linear kernel
        
            # Create triangular kernel with size 9 and peak at middle
            mid_point = (kernel_size - 1) // 2
            triangular_kernel = torch.zeros(
                kernel_size,
                device=disto_target.device,
                dtype=torch.float32
            )
            triangular_kernel[:mid_point+1] = torch.linspace(0, 1, mid_point+1)
            triangular_kernel[mid_point:] = torch.linspace(1, 0, kernel_size-mid_point)
            triangular_kernel = triangular_kernel / triangular_kernel.sum()  # Normalize
            triangular_kernel = triangular_kernel.view(1, 1, -1)
        
            b, l, _, n_bins = disto_target_logits.shape
            disto_target_logits_reshaped = disto_target_logits.view(
                b * l * l, 1, n_bins
            ).float()

            # Use reflect padding to handle boundaries correctly, avoiding wrap-around artifacts.
            # F.pad with mode='reflect' can fail on large tensors, so we implement it manually.
            padding_size = (kernel_size - 1) // 2
            if padding_size > 0:
                left_pad = disto_target_logits_reshaped[
                    :, :, 1 : 1 + padding_size
                ].flip(-1)
                right_pad = disto_target_logits_reshaped[
                    :, :, -1 - padding_size : -1
                ].flip(-1)
                padded_logits = torch.cat(
                    [left_pad, disto_target_logits_reshaped, right_pad], dim=-1
                )
            else:
                padded_logits = disto_target_logits_reshaped

            smoothed_disto_target_logits_reshaped = F.conv1d(
                padded_logits, triangular_kernel, padding="valid"
            )
            smoothed_disto_target_logits = (
                smoothed_disto_target_logits_reshaped.view(b, l, l, n_bins)
            )
            print("smoothed_disto_target_logits", smoothed_disto_target_logits)
            disto_target_logits = smoothed_disto_target_logits

            ############################ Normalize Distogram ############################

            # Normalize smoothed gt logits using offset and clamping
            gt_logits_mean = smoothed_disto_target_logits.mean()
            gt_logits_scale = smoothed_disto_target_logits.max() - smoothed_disto_target_logits.min()
            normalized_smoothed_disto_target_logits = pred_logits_mean + (smoothed_disto_target_logits - gt_logits_mean) * (distogram_scale / gt_logits_scale)

            print("normalized_smoothed_disto_target_logits", normalized_smoothed_disto_target_logits)

            disto_target_logits = torch.clamp(normalized_smoothed_disto_target_logits, min=pred_logits_min, max=pred_logits_max)

            print("final disto_target_logits", disto_target_logits)
            print("final disto_target_logits mean", disto_target_logits.mean())

            ############################ Extract Pair Features from Distogram ############################

            distogram_linear = self.distogram_module.distogram
            disto_target_logits_no_bias = (
                disto_target_logits - distogram_linear.bias.detach()
            )
            gt_z = (
                torch.einsum(
                    "bklj, ij->bkli",
                    disto_target_logits_no_bias,
                    torch.linalg.pinv(distogram_linear.weight.detach()),
                )
                / 2
            )
            print("z shape", z.shape)
            print("gt_z shape", gt_z.shape)
            print("symmetric gt_z", torch.allclose(gt_z, gt_z.transpose(1, 2)))
            # assert torch.allclose(
            #     torch.argmax(self.distogram_module(gt_z).float()[:, :, :, 0], dim=-1),
            #     torch.argmax(disto_target_logits.float(), dim=-1),
            #     atol=1e-4,
            # ), f"gt_z: {torch.argmax(self.distogram_module(gt_z).float()[:, :, :, 0], dim=-1)}, disto_target_logits: {torch.argmax(disto_target_logits.float(), dim=-1)}"

        if (
            self.run_trunk_and_structure
            and ((not self.training) or self.confidence_prediction)
            and (not self.skip_run_structure)
        ):
            diffusion_conditioning_list = []

            for i, (s, z) in enumerate(zip(s_list, z_list)):
                print("diffusion conditioning sample", i)
                
                if self.checkpoint_diffusion_conditioning and self.training:
                    # TODO decide whether this should be with bf16 or not
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        torch.utils.checkpoint.checkpoint(
                            self.diffusion_conditioning,
                            s,
                            z if not self.predict_args.get("structure_ablation", False) else gt_z,
                            relative_position_encoding,
                            feats,
                        )
                    )
                else:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        self.diffusion_conditioning(
                            s_trunk=s,
                            z_trunk=z if not self.predict_args.get("structure_ablation", False) else gt_z,
                            relative_position_encoding=relative_position_encoding,
                            feats=feats,
                        )
                    )
                # print("q shape", q.shape)
                # print("c shape", c.shape)
                # print("atom_enc_bias shape", atom_enc_bias.shape)
                # print("atom_dec_bias shape", atom_dec_bias.shape)
                # print("token_trans_bias shape", token_trans_bias.shape)
                diffusion_conditioning = {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }
                diffusion_conditioning_list.append(diffusion_conditioning)

            if self.steering_args.get("use_confidence", False) or self.logmd_args.get("logmd_confidence", False) or self.logmd_args.get("save_intermediate_confidence", False):
                confidence_kwargs = {
                    "confidence_module": self.confidence_module,
                    "pred_distogram_logits": pdistogram.detach(),
                    "run_sequentially": run_confidence_sequentially,
                    "z_trunk": z_list if not self.predict_args.get("structure_ablation", False) else gt_z,
                }
                if self.steering_args.get("use_boltz1_confidence_steering", False):
                    b1_model = self.boltz1_model_container[0]
                    b1_model = b1_model.to(dtype=self.dtype)

                    # Optionally add Boltz1 trunk features
                    if self.steering_args.get("use_boltz1_trunk_features", False):
                        # Move model to GPU and set correct dtype before use
                        b1_model.to(s_inputs.device)
                        s_inputs_b1, s_b1, z_b1, _, _ = b1_model.trunk_forward(
                            feats["boltz1_feats"], recycling_steps
                        )
                        confidence_kwargs["boltz1_trunk_features"] = {
                            "s_inputs": s_inputs_b1.to(self.dtype),
                            "s": s_b1.to(self.dtype),
                            "z": z_b1.to(self.dtype),
                        }
                        b1_model.to("cpu") # save memory
                        torch.cuda.empty_cache()
                    # Always hand over the Boltz1 confidence module
                    boltz1_conf_mod = self.boltz1_model_container[0].confidence_module
                    boltz1_conf_mod.to(s_inputs.device)
                    boltz1_conf_mod.use_s_diffusion = False # token rep of Boltz1 and 2 is different so s_diffusion also different
                    confidence_kwargs["boltz1_confidence_module"] = boltz1_conf_mod
            else:
                confidence_kwargs = None

            if max_multiplicity is None or max_multiplicity > diffusion_samples:
                max_multiplicity = diffusion_samples
            merged_struct_out = None
            with torch.autocast("cuda", enabled=False):
                for i in range(math.ceil(diffusion_samples / max_multiplicity)):
                    struct_out = self.structure_module.sample(
                        s_trunk=[s.float() for s in s_list],
                        s_inputs=s_inputs.float(),
                        feats=feats,
                        num_sampling_steps=num_sampling_steps,
                        atom_mask=feats["atom_pad_mask"].float(),
                        multiplicity=max_multiplicity if i < diffusion_samples // max_multiplicity else diffusion_samples % max_multiplicity,
                        max_parallel_samples=max_parallel_samples,
                        steering_args=self.steering_args,
                        logmd_args=self.logmd_args,
                        diffusion_conditioning=diffusion_conditioning_list,
                        confidence_kwargs=confidence_kwargs,
                        superposition=self.predict_args.get("superposition", True),
                    )
                    if merged_struct_out is None:
                        merged_struct_out = struct_out
                    else:
                        for k, v in struct_out.items():
                            if v is not None:
                                merged_struct_out[k] = torch.cat([merged_struct_out[k].cpu(), v.cpu()], dim=0) # save memory for large diffusion_samples
                dict_out.update(merged_struct_out)

            print("Struture Prediction Done")

            if self.predict_bfactor:
                pbfactor = self.bfactor_module(s)
                dict_out["pbfactor"] = pbfactor

        if self.training and self.confidence_prediction:
            assert len(feats["coords"].shape) == 4
            assert (
                feats["coords"].shape[1] == 1
            )  # Only one conformation is supported for confidence

        # Compute structure module
        if self.training and self.structure_prediction_training:
            assert s_z_samples == 1, "Structure Training only supports 1 sample"

            atom_coords = feats["coords"]
            B, K, L = atom_coords.shape[0:3]
            assert K in (
                multiplicity_diffusion_train,
                1,
            )  # TODO make check somewhere else, expand to m % N == 0, m > N
            atom_coords = atom_coords.reshape(B * K, L, 3)
            atom_coords = atom_coords.repeat_interleave(
                multiplicity_diffusion_train // K, 0
            )
            feats["coords"] = atom_coords  # (multiplicity, L, 3)
            assert len(feats["coords"].shape) == 3

            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module(
                    s_trunk=s_list[0].float(),
                    s_inputs=s_inputs.float(),
                    feats=feats,
                    multiplicity=multiplicity_diffusion_train,
                    diffusion_conditioning=diffusion_conditioning,
                )
                dict_out.update(struct_out)

        elif self.training:
            feats["coords"] = feats["coords"].squeeze(1)
            assert len(feats["coords"].shape) == 3

        if self.confidence_prediction:
            print("Boltz2 Confidence Calculation")
            confidence_outputs = []
            for i, (s, z) in enumerate(zip(s_list, z_list)):
                confidence_out = self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=(
                        dict_out["pdistogram"][
                            :, :, :, i
                        ].detach()  # TODO only implemeted for 1 distogram
                    ),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
                def move_to_cpu(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu()
                    elif isinstance(obj, dict):
                        return {k: move_to_cpu(v) for k, v in obj.items()}
                    return obj
                confidence_outputs.append(move_to_cpu(confidence_out))

            if confidence_outputs:
                mean_confidence_out = {}
                num_samples = len(confidence_outputs)
                if num_samples > 0:
                    for key in confidence_outputs[0]:
                        if key == "pair_chains_iptm":
                            mean_confidence_out[key] = {}
                            for idx1 in confidence_outputs[0][key]:
                                mean_confidence_out[key][idx1] = {}
                                for idx2 in confidence_outputs[0][key][idx1]:
                                    tensors_to_stack = [
                                        co[key][idx1][idx2]
                                        for co in confidence_outputs
                                    ]
                                    mean_confidence_out[key][idx1][idx2] = (
                                        torch.stack(tensors_to_stack).mean(dim=0)
                                    )
                        else:
                            tensors_to_stack = [
                                co[key] for co in confidence_outputs
                            ]
                            mean_confidence_out[key] = torch.stack(
                                tensors_to_stack
                            ).mean(dim=0)
                    dict_out.update(mean_confidence_out)

            if self.steering_args.get("use_boltz1_confidence_steering", False):
                print("Boltz1 Confidence Calculation")

                b1_model = self.boltz1_model_container[0]
                b1_model = b1_model.to(dtype=self.dtype)
                
                if self.steering_args.get("use_boltz1_trunk_features", False):
                    if confidence_kwargs is not None:
                        bl_feats = confidence_kwargs["boltz1_trunk_features"]
                        s_inputs_b1 = bl_feats["s_inputs"]
                        s_b1 = bl_feats["s"]
                        z_b1 = bl_feats["z"]
                    else:
                        b1_model.to(s_inputs.device)
                        s_inputs_b1, s_b1, z_b1, _, _ = b1_model.trunk_forward(
                            feats["boltz1_feats"], recycling_steps
                        )
                        s_inputs_b1 = s_inputs_b1.to(self.dtype)
                        s_b1 = s_b1.to(self.dtype)
                        z_b1 = z_b1.to(self.dtype)
                        b1_model.to("cpu")
                        torch.cuda.empty_cache()
                else:
                    b1_feats = confidence_kwargs["boltz1_trunk_features"]
                    s_inputs_b1 = s_inputs
                    s_b1 = s
                    z_b1 = z
                boltz1_conf_mod = self.boltz1_model_container[0].confidence_module
                boltz1_conf_mod.to(s_inputs.device)
                boltz1_conf_mod.use_s_diffusion = False # token rep of Boltz1 and 2 is different so s_diffusion also different
                boltz1_confidence_out = boltz1_conf_mod(
                    s_inputs=s_inputs_b1,
                    s=s_b1,
                    z=z_b1,
                    s_diffusion=None,
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats["boltz1_feats"] if self.steering_args.get("use_boltz1_trunk_features", False) else feats,
                    pred_distogram_logits=(
                        dict_out["pdistogram"][
                            :, :, :, :
                        ].mean(dim=3).detach()  # mean over pd samples
                    ),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
                dict_out.update({f"{k}_boltz1": v for k, v in boltz1_confidence_out.items()})

        if (
            self.predict_args.get("confidence_ablation", False)
            and not self.training
            and self.confidence_prediction
        ):
            assert s_z_samples == 1, "Confidence ablation only supports 1 sample"
            
            # Predicted coords
            pred_coords = dict_out["sample_atom_coords"].detach()
            print("pred_coords shape", pred_coords.shape)
            print("pred_coords", pred_coords)

            # Ground truth coords
            gt_coords = feats["coords"].squeeze(1)  # B, L, 3
            print("gt_coords shape", gt_coords.shape)
            print("gt_coords", gt_coords)

            # Ablation cases
            # 1. pred_disto + gt_coords
            ablation_pd_gtc = self.confidence_module(
                s_inputs=s_inputs.detach(),
                s=s.detach(),
                z=z.detach(),
                x_pred=gt_coords.repeat(diffusion_samples, 1, 1).contiguous(),
                feats=feats,
                pred_distogram_logits=pred_distogram_logits.repeat(
                    diffusion_samples, 1, 1, 1
                ).contiguous(),
                multiplicity=diffusion_samples,
                run_sequentially=run_confidence_sequentially,
                use_kernels=self.use_kernels,
            )
            dict_out.update({f"ablation_pd_gtc_{k}": v for k, v in ablation_pd_gtc.items()})

            # 2. gt_disto + pred_coords
            ablation_gtd_pc = self.confidence_module(
                s_inputs=s_inputs.detach(),
                s=s.detach(),
                z=gt_z.detach(),
                x_pred=pred_coords,
                feats=feats,
                pred_distogram_logits=disto_target_logits.contiguous(),
                multiplicity=diffusion_samples,
                run_sequentially=run_confidence_sequentially,
                use_kernels=self.use_kernels,
            )
            dict_out.update({f"ablation_gtd_pc_{k}": v for k, v in ablation_gtd_pc.items()})

            if self.predict_args.get("confidence_ablation_all_gt", False):
                # 3. gt_disto + gt_coords
                ablation_gtd_gtc = self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=gt_z.detach(),
                    x_pred=gt_coords.repeat(diffusion_samples, 1, 1).contiguous(),
                    feats=feats,
                    pred_distogram_logits=disto_target_logits.contiguous(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
                dict_out.update(
                    {f"ablation_gtd_gtc_{k}": v for k, v in ablation_gtd_gtc.items()}
                )

        if self.affinity_prediction:
            assert s_z_samples == 1, "Affinity prediction only supports 1 sample"

            pad_token_mask = feats["token_pad_mask"][0]
            rec_mask = feats["mol_type"][0] == 0
            rec_mask = rec_mask * pad_token_mask
            lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
            lig_mask = lig_mask * pad_token_mask
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )
            z_affinity = z * cross_pair_mask[None, :, :, None]

            argsort = torch.argsort(dict_out["iptm"], descending=True)
            best_idx = argsort[0].item()
            coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][
                None, None
            ]
            s_inputs = self.input_embedder(feats, affinity=True)

            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    dict_out_affinity1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )

                    dict_out_affinity1["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity1["affinity_logits_binary"]
                        )
                    )
                    dict_out_affinity2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity2["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity2["affinity_logits_binary"]
                        )
                    )

                    dict_out_affinity_ensemble = {
                        "affinity_pred_value": (
                            dict_out_affinity1["affinity_pred_value"]
                            + dict_out_affinity2["affinity_pred_value"]
                        )
                        / 2,
                        "affinity_probability_binary": (
                            dict_out_affinity1["affinity_probability_binary"]
                            + dict_out_affinity2["affinity_probability_binary"]
                        )
                        / 2,
                    }

                    dict_out_affinity1 = {
                        "affinity_pred_value1": dict_out_affinity1[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary1": dict_out_affinity1[
                            "affinity_probability_binary"
                        ],
                    }
                    dict_out_affinity2 = {
                        "affinity_pred_value2": dict_out_affinity2[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary2": dict_out_affinity2[
                            "affinity_probability_binary"
                        ],
                    }
                    if self.affinity_mw_correction:
                        model_coef = 1.03525938
                        mw_coef = -0.59992683
                        bias = 2.83288489
                        mw = feats["affinity_mw"][0] ** 0.3
                        dict_out_affinity_ensemble["affinity_pred_value"] = (
                            model_coef
                            * dict_out_affinity_ensemble["affinity_pred_value"]
                            + mw_coef * mw
                            + bias
                        )

                    dict_out.update(dict_out_affinity_ensemble)
                    dict_out.update(dict_out_affinity1)
                    dict_out.update(dict_out_affinity2)
                else:
                    dict_out_affinity = self.affinity_module(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out.update(
                        {
                            "affinity_pred_value": dict_out_affinity[
                                "affinity_pred_value"
                            ],
                            "affinity_probability_binary": torch.nn.functional.sigmoid(
                                dict_out_affinity["affinity_logits_binary"]
                            ),
                        }
                    )

        return dict_out

    def get_true_coordinates(
        self,
        batch: dict[str, Tensor],
        out: dict[str, Tensor],
        diffusion_samples: int,
        symmetry_correction: bool,
        expand_to_diffusion_samples: bool = True,
    ):
        if symmetry_correction:
            msg = "expand_to_diffusion_samples must be true for symmetry correction."
            assert expand_to_diffusion_samples, msg

        return_dict = {}

        assert (
            batch["coords"].shape[0] == 1
        ), f"Validation is not supported for batch sizes={batch['coords'].shape[0]}"

        if symmetry_correction:
            true_coords = []
            true_coords_resolved_mask = []
            for idx in range(batch["token_index"].shape[0]):
                for rep in range(diffusion_samples):
                    i = idx * diffusion_samples + rep
                    best_true_coords, best_true_coords_resolved_mask = (
                        minimum_lddt_symmetry_coords(
                            coords=out["sample_atom_coords"][i : i + 1],
                            feats=batch,
                            index_batch=idx,
                        )
                    )
                    true_coords.append(best_true_coords)
                    true_coords_resolved_mask.append(best_true_coords_resolved_mask)

            true_coords = torch.cat(true_coords, dim=0)
            true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
            true_coords = true_coords.unsqueeze(1)

            true_coords_resolved_mask = true_coords_resolved_mask

            return_dict["true_coords"] = true_coords
            return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
            return_dict["rmsds"] = 0
            return_dict["best_rmsd_recall"] = 0

        else:
            K, L = batch["coords"].shape[1:3]

            true_coords_resolved_mask = batch["atom_resolved_mask"]
            true_coords = batch["coords"].squeeze(0)
            if expand_to_diffusion_samples:
                true_coords = true_coords.repeat((diffusion_samples, 1, 1)).reshape(
                    diffusion_samples, K, L, 3
                )

                true_coords_resolved_mask = true_coords_resolved_mask.repeat_interleave(
                    diffusion_samples, dim=0
                )  # since all masks are the same across conformers and diffusion samples, can just repeat S times
            else:
                true_coords_resolved_mask = true_coords_resolved_mask.squeeze(0)

            return_dict["true_coords"] = true_coords
            return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
            return_dict["rmsds"] = 0
            return_dict["best_rmsd_recall"] = 0
            return_dict["best_rmsd_precision"] = 0

        return return_dict

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        # Sample recycling steps
        if self.no_random_recycling_training:
            recycling_steps = self.training_args.recycling_steps
        else:
            rgn = np.random.default_rng(self.global_step)
            recycling_steps = rgn.integers(
                0, self.training_args.recycling_steps + 1
            ).item()

        if self.training_args.get("sampling_steps_random", None) is not None:
            rgn_samplng_steps = np.random.default_rng(self.global_step)
            sampling_steps = rgn_samplng_steps.choice(
                self.training_args.sampling_steps_random
            )
        else:
            sampling_steps = self.training_args.sampling_steps

        # Compute the forward pass
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity,
            diffusion_samples=self.training_args.diffusion_samples,
        )

        # Compute losses
        if self.structure_prediction_training:
            disto_loss, _ = distogram_loss(
                out,
                batch,
                aggregate_distogram=self.aggregate_distogram,
            )
            try:
                diffusion_loss_dict = self.structure_module.compute_loss(
                    batch,
                    out,
                    multiplicity=self.training_args.diffusion_multiplicity,
                    **self.diffusion_loss_args,
                )
            except Exception as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                return None

            if self.predict_bfactor:
                bfactor_loss = bfactor_loss_fn(out, batch)
            else:
                bfactor_loss = 0.0

        else:
            disto_loss = 0.0
            bfactor_loss = 0.0
            diffusion_loss_dict = {"loss": 0.0, "loss_breakdown": {}}

        if self.confidence_prediction:
            try:
                # confidence model symmetry correction
                return_dict = self.get_true_coordinates(
                    batch,
                    out,
                    diffusion_samples=self.training_args.diffusion_samples,
                    symmetry_correction=self.training_args.symmetry_correction,
                )
            except Exception as e:
                print(f"Skipping batch with id {batch['pdb_id']} due to error: {e}")
                return None

            true_coords = return_dict["true_coords"]
            true_coords_resolved_mask = return_dict["true_coords_resolved_mask"]

            # TODO remove once multiple conformers are supported
            K = true_coords.shape[1]
            assert (
                K == 1
            ), f"Confidence_prediction is not supported for num_ensembles_val={K}."

            # For now, just take the only conformer.
            true_coords = true_coords.squeeze(1)  # (S, L, 3)
            batch["frames_idx"] = batch["frames_idx"].squeeze(
                1
            )  # remove conformer dimension
            batch["frame_resolved_mask"] = batch["frame_resolved_mask"].squeeze(
                1
            )  # remove conformer dimension

            confidence_loss_dict = confidence_loss(
                out,
                batch,
                true_coords,
                true_coords_resolved_mask,
                token_level_confidence=self.token_level_confidence,
                alpha_pae=self.alpha_pae,
                multiplicity=self.training_args.diffusion_samples,
            )

        else:
            confidence_loss_dict = {
                "loss": torch.tensor(0.0, device=batch["token_index"].device),
                "loss_breakdown": {},
            }

        # Aggregate losses
        # NOTE: we already have an implicit weight in the losses induced by dataset sampling
        # NOTE: this logic works only for datasets with confidence labels
        loss = (
            self.training_args.confidence_loss_weight * confidence_loss_dict["loss"]
            + self.training_args.diffusion_loss_weight * diffusion_loss_dict["loss"]
            + self.training_args.distogram_loss_weight * disto_loss
            + self.training_args.get("bfactor_loss_weight", 0.0) * bfactor_loss
        )

        if not (self.global_step % self.log_loss_every_steps):
            # Log losses
            if self.validate_structure:
                self.log("train/distogram_loss", disto_loss)
                self.log("train/diffusion_loss", diffusion_loss_dict["loss"])
                for k, v in diffusion_loss_dict["loss_breakdown"].items():
                    self.log(f"train/{k}", v)

            if self.confidence_prediction:
                self.train_confidence_loss_logger.update(
                    confidence_loss_dict["loss"].detach()
                )
                for k in self.train_confidence_loss_dict_logger:
                    self.train_confidence_loss_dict_logger[k].update(
                        (
                            confidence_loss_dict["loss_breakdown"][k].detach()
                            if torch.is_tensor(
                                confidence_loss_dict["loss_breakdown"][k]
                            )
                            else confidence_loss_dict["loss_breakdown"][k]
                        )
                    )
            self.log("train/loss", loss)
            self.training_log()
        return loss

    def training_log(self):
        self.log("train/grad_norm", self.gradient_norm(self), prog_bar=False)
        self.log("train/param_norm", self.parameter_norm(self), prog_bar=False)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

        self.log(
            "train/param_norm_msa_module",
            self.parameter_norm(self.msa_module),
            prog_bar=False,
        )

        self.log(
            "train/param_norm_pairformer_module",
            self.parameter_norm(self.pairformer_module),
            prog_bar=False,
        )

        self.log(
            "train/param_norm_structure_module",
            self.parameter_norm(self.structure_module),
            prog_bar=False,
        )

        if self.confidence_prediction:
            self.log(
                "train/grad_norm_confidence_module",
                self.gradient_norm(self.confidence_module),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_confidence_module",
                self.parameter_norm(self.confidence_module),
                prog_bar=False,
            )

    def on_train_epoch_end(self):
        if self.confidence_prediction:
            self.log(
                "train/confidence_loss",
                self.train_confidence_loss_logger,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            for k, v in self.train_confidence_loss_dict_logger.items():
                self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)

    def gradient_norm(self, module):
        parameters = [
            p.grad.norm(p=2) ** 2
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def parameter_norm(self, module):
        parameters = [p.norm(p=2) ** 2 for p in module.parameters() if p.requires_grad]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        if self.validate_structure:
            try:
                msg = "Only batch=1 is supported for validation"
                assert batch["idx_dataset"].shape[0] == 1, msg

                # Select validator based on dataset
                idx_dataset = batch["idx_dataset"][0].item()
                validator = self.validator_mapper[idx_dataset]

                # Run forward pass
                out = validator.run_model(
                    model=self, batch=batch, idx_dataset=idx_dataset
                )
                # Compute validation step
                validator.process(
                    model=self, batch=batch, out=out, idx_dataset=idx_dataset
                )
            except RuntimeError as e:  # catch out of memory exceptions
                idx_dataset = batch["idx_dataset"][0].item()
                if "out of memory" in str(e):
                    msg = f"| WARNING: ran out of memory, skipping batch, {idx_dataset}"
                    print(msg)
                    torch.cuda.empty_cache()
                    gc.collect()
                    return
                raise e
        else:
            try:
                out = self(
                    batch,
                    recycling_steps=self.validation_args.recycling_steps,
                    num_sampling_steps=self.validation_args.sampling_steps,
                    diffusion_samples=self.validation_args.diffusion_samples,
                    run_confidence_sequentially=self.validation_args.get(
                        "run_confidence_sequentially", False
                    ),
                )
            except RuntimeError as e:  # catch out of memory exceptions
                idx_dataset = batch["idx_dataset"][0].item()
                if "out of memory" in str(e):
                    msg = f"| WARNING: ran out of memory, skipping batch, {idx_dataset}"
                    print(msg)
                    torch.cuda.empty_cache()
                    gc.collect()
                    return
                raise e

    def on_validation_epoch_end(self):
        """Aggregate all metrics for each validator."""
        if self.validate_structure:
            for validator in self.validator_mapper.values():
                # This will aggregate, compute and log all metrics
                validator.on_epoch_end(model=self)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        try:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                s_z_samples=self.predict_args["s_z_samples"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                max_parallel_samples=self.predict_args["max_parallel_samples"],
                max_multiplicity=self.predict_args["max_multiplicity"],
                run_confidence_sequentially=True,
            )
            pred_dict = {"exception": False}
            if "keys_dict_batch" in self.predict_args:
                for key in self.predict_args["keys_dict_batch"]:
                    pred_dict[key] = batch[key]

            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["token_masks"] = batch["token_pad_mask"]

            if "keys_dict_out" in self.predict_args:
                for key in self.predict_args["keys_dict_out"]:
                    pred_dict[key] = out[key]
            pred_dict["coords"] = out["sample_atom_coords"]
            if "pdistogram" in out:
                pred_dict["pdistogram"] = out["pdistogram"]
            if self.confidence_prediction:
                # pred_dict["confidence"] = out.get("ablation_confidence", None)
                pred_dict["pde"] = out["pde"]
                pred_dict["plddt"] = out["plddt"]
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"]
                    + (
                        out["iptm"]
                        if not torch.allclose(
                            out["iptm"], torch.zeros_like(out["iptm"])
                        )
                        else out["ptm"]
                    )
                ) / 5

                pred_dict["complex_plddt"] = out["complex_plddt"]
                pred_dict["complex_iplddt"] = out["complex_iplddt"]
                pred_dict["complex_pde"] = out["complex_pde"]
                pred_dict["complex_ipde"] = out["complex_ipde"]
                if self.alpha_pae > 0:
                    pred_dict["pae"] = out["pae"]
                    pred_dict["ptm"] = out["ptm"]
                    pred_dict["iptm"] = out["iptm"]
                    pred_dict["ligand_iptm"] = out["ligand_iptm"]
                    pred_dict["protein_iptm"] = out["protein_iptm"]
                    pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]
                    # pred_dict["ptm_energy"] = out["ptm_energy"]  # Not returned by confidence_v2.py
                    # pred_dict["iptm_energy"] = out["iptm_energy"]  # Not returned by confidence_v2.py
                if "plddt_boltz1" in out:
                    pred_dict["pde_boltz1"] = out["pde_boltz1"]
                    pred_dict["plddt_boltz1"] = out["plddt_boltz1"]
                    pred_dict["complex_plddt_boltz1"] = out["complex_plddt_boltz1"]
                    pred_dict["complex_iplddt_boltz1"] = out["complex_iplddt_boltz1"]
                    pred_dict["complex_pde_boltz1"] = out["complex_pde_boltz1"]
                    pred_dict["complex_ipde_boltz1"] = out["complex_ipde_boltz1"]
                    if self.alpha_pae > 0:
                        pred_dict["pae_boltz1"] = out["pae_boltz1"]
                        pred_dict["ptm_boltz1"] = out["ptm_boltz1"]
                        pred_dict["iptm_boltz1"] = out["iptm_boltz1"]
                        pred_dict["ligand_iptm_boltz1"] = out["ligand_iptm_boltz1"]
                        pred_dict["protein_iptm_boltz1"] = out["protein_iptm_boltz1"]
                        pred_dict["pair_chains_iptm_boltz1"] = out["pair_chains_iptm_boltz1"]
                        # Energy fields intentionally omitted for v2
            if self.affinity_prediction:
                pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
                pred_dict["affinity_probability_binary"] = out[
                    "affinity_probability_binary"
                ]
                if self.affinity_ensemble:
                    pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
                    pred_dict["affinity_probability_binary1"] = out[
                        "affinity_probability_binary1"
                    ]
                    pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
                    pred_dict["affinity_probability_binary2"] = out[
                        "affinity_probability_binary2"
                    ]
            if self.predict_args.get("confidence_ablation", False):
                for prefix in ["ablation_pd_gtc_", "ablation_gtd_pc_"]:
                    if f"{prefix}plddt" in out:
                        pred_dict[f"{prefix}plddt"] = out[f"{prefix}plddt"]
                        pred_dict[f"{prefix}pde"] = out[f"{prefix}pde"]
                        pred_dict[f"{prefix}pae"] = out[f"{prefix}pae"]
                        pred_dict[f"{prefix}ptm"] = out[f"{prefix}ptm"]
                        pred_dict[f"{prefix}iptm"] = out[f"{prefix}iptm"]

            if self.predict_args.get("confidence_ablation_all_gt", False):
                prefix = "ablation_gtd_gtc_"
                if f"{prefix}plddt" in out:
                    pred_dict[f"{prefix}plddt"] = out[f"{prefix}plddt"]
                    pred_dict[f"{prefix}pde"] = out[f"{prefix}pde"]
                    pred_dict[f"{prefix}pae"] = out[f"{prefix}pae"]
                    pred_dict[f"{prefix}ptm"] = out[f"{prefix}ptm"]
                    pred_dict[f"{prefix}iptm"] = out[f"{prefix}iptm"]

            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise {"exception": True}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        param_dict = dict(self.named_parameters())

        if self.structure_prediction_training:
            all_parameter_names = [
                pn for pn, p in self.named_parameters() if p.requires_grad
            ]
        else:
            all_parameter_names = [
                pn
                for pn, p in self.named_parameters()
                if p.requires_grad
                and ("out_token_feat_update" in pn or "confidence_module" in pn)
            ]

        if self.training_args.get("weight_decay", 0.0) > 0:
            w_decay = self.training_args.get("weight_decay", 0.0)
            if self.training_args.get("weight_decay_exclude", False):
                nodecay_params_names = [
                    pn
                    for pn in all_parameter_names
                    if (
                        "norm" in pn
                        or "rel_pos" in pn
                        or ".s_init" in pn
                        or ".z_init_" in pn
                        or "token_bonds" in pn
                        or "embed_atom_features" in pn
                        or "dist_bin_pairwise_embed" in pn
                    )
                ]
                nodecay_params = [param_dict[pn] for pn in nodecay_params_names]
                decay_params = [
                    param_dict[pn]
                    for pn in all_parameter_names
                    if pn not in nodecay_params_names
                ]
                optim_groups = [
                    {"params": decay_params, "weight_decay": w_decay},
                    {"params": nodecay_params, "weight_decay": 0.0},
                ]
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    betas=(
                        self.training_args.adam_beta_1,
                        self.training_args.adam_beta_2,
                    ),
                    eps=self.training_args.adam_eps,
                    lr=self.training_args.base_lr,
                )

            else:
                optimizer = torch.optim.AdamW(
                    [param_dict[pn] for pn in all_parameter_names],
                    betas=(
                        self.training_args.adam_beta_1,
                        self.training_args.adam_beta_2,
                    ),
                    eps=self.training_args.adam_eps,
                    lr=self.training_args.base_lr,
                    weight_decay=self.training_args.get("weight_decay", 0.0),
                )
        else:
            optimizer = torch.optim.AdamW(
                [param_dict[pn] for pn in all_parameter_names],
                betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
                eps=self.training_args.adam_eps,
                lr=self.training_args.base_lr,
                weight_decay=self.training_args.get("weight_decay", 0.0),
            )

        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Ignore the lr from the checkpoint
        lr = self.training_args.max_lr
        weight_decay = self.training_args.weight_decay
        if "optimizer_states" in checkpoint:
            for state in checkpoint["optimizer_states"]:
                for group in state["param_groups"]:
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay
        if "lr_schedulers" in checkpoint:
            for scheduler in checkpoint["lr_schedulers"]:
                scheduler["max_lr"] = lr
                scheduler["base_lrs"] = [lr] * len(scheduler["base_lrs"])
                scheduler["_last_lr"] = [lr] * len(scheduler["_last_lr"])

        # Ignore the training diffusion_multiplicity and recycling steps from the checkpoint
        if "hyper_parameters" in checkpoint:
            checkpoint["hyper_parameters"]["training_args"]["max_lr"] = lr
            checkpoint["hyper_parameters"]["training_args"][
                "diffusion_multiplicity"
            ] = self.training_args.diffusion_multiplicity
            checkpoint["hyper_parameters"]["training_args"]["recycling_steps"] = (
                self.training_args.recycling_steps
            )
            checkpoint["hyper_parameters"]["training_args"]["weight_decay"] = (
                self.training_args.weight_decay
            )

    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks.

        Returns
        -------
        List[Callback]
            List of callbacks to be used in the model.

        """
        return [EMA(self.ema_decay)] if self.use_ema else []

    def trunk_forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
    ):
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence embeddings
            s_init = self.s_init(s_inputs)

            # Initialize pairwise embeddings
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
            z_init = z_init + self.contact_conditioning(feats)

            if not self.is_in_dataparallel and next(self.pairformer_module.parameters()).device != s_init.device:
                self.pairformer_module = self.pairformer_module.to(s_init.device)

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]
            if self.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        self.training
                        and self.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        # Issue with unused parameters in autocast
                        if (
                            self.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Apply recycling
                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))

                        # Compute pairwise stack
                        if self.use_templates:
                            if self.is_template_compiled and not self.training:
                                template_module = self.template_module._orig_mod  # noqa: SLF001
                            else:
                                template_module = self.template_module

                            z = z + template_module(
                                z, feats, pair_mask, use_kernels=self.use_kernels
                            )

                        if self.is_msa_compiled and not self.training:
                            msa_module = self.msa_module._orig_mod  # noqa: SLF001
                        else:
                            msa_module = self.msa_module

                        z = z + msa_module(
                            z, s_inputs, feats, use_kernels=self.use_kernels, use_dropout=self.use_dropout
                        )

                        # Revert to uncompiled version for validation
                        if self.is_pairformer_compiled and not self.training:
                            pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                        else:
                            pairformer_module = self.pairformer_module

                        s, z = pairformer_module(
                            s,
                            z,
                            mask=mask,
                            pair_mask=pair_mask,
                            use_kernels=self.use_kernels,
                            use_dropout=self.use_dropout,
                        )
                
                if not self.training and not self.is_in_dataparallel:
                    self.msa_module = self.msa_module.cpu()
                    if self.use_templates:
                        self.template_module = self.template_module.cpu()
                    gc.collect()
                    torch.cuda.empty_cache()

        return s_inputs, s, z, relative_position_encoding
