# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
import random
import json

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.types import StructureV2
from boltz.model.loss.diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encoders import FourierEmbedding
from boltz.model.modules.transformers import ConditionedTransitionBlock
from boltz.model.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from boltz.model.potentials.potentials import get_potentials
from boltz.model.layers.confidence_utils import compute_ptms

from boltz.data.write.pdb import to_pdb


class OutTokenFeatUpdate(Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            # post_layer_norm=transformer_post_ln,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            # transformer_post_layer_norm=transformer_post_ln,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return {"r_update": r_update, "token_a": a.detach()}


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        accumulate_token_repr: bool = False,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.accumulate_token_repr = accumulate_token_repr
        if self.accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
        step_scale: float = None,
        sigma_next: float = None,
        ll: torch.Tensor = None,
        superposition: bool = True,
        verbose: bool = False,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        if isinstance(network_condition_kwargs["s_trunk"], list):
            assert isinstance(network_condition_kwargs["diffusion_conditioning"], list)
            assert len(network_condition_kwargs["s_trunk"]) == len(network_condition_kwargs["diffusion_conditioning"])

            # kappa = torch.ones_like(vel, device=vel.device) / vel.shape[0] # Naive Implementation
            if superposition:
                kappa = torch.softmax(ll, dim=0)
                kappa[kappa < 1e-4] = 0 # effective kappa
            else:
                kappa = torch.eye(ll.shape[0], device=ll.device)
            if verbose:
                print("kappa", kappa)

            vel_list = []
            for condition_idx, (s_trunk, diffusion_conditioning) in enumerate(zip(network_condition_kwargs["s_trunk"], network_condition_kwargs["diffusion_conditioning"])):
                if superposition:
                    particle_idx = torch.where(kappa[condition_idx] > 0)[0]
                else:
                    particle_idx = torch.tensor([condition_idx]).to(noised_atom_coords.device)
                if len(particle_idx) == 0:
                    if verbose:
                        print("No particles for condition", condition_idx)
                    vel = torch.zeros_like(noised_atom_coords)
                    vel_list.append(vel)
                    continue
                net_out = self.score_model(
                    r_noisy=self.c_in(padded_sigma[particle_idx]) * noised_atom_coords[particle_idx],
                    times=self.c_noise(sigma[particle_idx]),
                    s_trunk=s_trunk,
                    diffusion_conditioning=diffusion_conditioning,
                    s_inputs=network_condition_kwargs["s_inputs"],
                    feats=network_condition_kwargs["feats"],
                    multiplicity=len(particle_idx),
                )
                r_update = net_out["r_update"]
                denoised_coords = (
                    self.c_skip(padded_sigma[particle_idx]) * noised_atom_coords[particle_idx]
                    + self.c_out(padded_sigma[particle_idx]) * r_update
                )
                vel = torch.zeros_like(noised_atom_coords)
                vel[particle_idx] = (noised_atom_coords[particle_idx] - denoised_coords) / padded_sigma[particle_idx]
                vel_list.append(vel)

            vel = torch.stack(vel_list, dim=0)
            kappa = rearrange(kappa, "c b  -> c b 1 1")
            vel_composed = (kappa * vel).sum(dim=0)
            denoised_coords = noised_atom_coords - padded_sigma * vel_composed

            dsigma = sigma_next - sigma
            padded_dsigma = rearrange(dsigma, "b -> b 1 1")
            dx = padded_dsigma * step_scale * vel_composed
            if verbose:
                print("dx norm", (dx**2).mean().sqrt().item())
            if superposition: # ll is only used for superposition
                ll = ll - (step_scale * vel * (dx.unsqueeze(0) + padded_dsigma.unsqueeze(0) * step_scale * vel) / padded_sigma.unsqueeze(0)).sum((-2, -1))
            return denoised_coords, net_out["token_a"], ll, vel
        else:
            net_out = self.score_model(
                r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
                times=self.c_noise(sigma),
                **network_condition_kwargs,
            )
            r_update = net_out["r_update"]

            denoised_coords = (
                self.c_skip(padded_sigma) * noised_atom_coords
                + self.c_out(padded_sigma) * r_update
            )
            return denoised_coords, net_out["token_a"]

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        logmd_args=None,
        confidence_kwargs=None,
        superposition=True,
        verbose=False,
        **network_condition_kwargs,
    ):
        structure = None
        if steering_args.get("use_openmm_energy", False) or \
           steering_args.get("use_rosetta_energy", False):
            structure = StructureV2.load(steering_args["targets_dir"] / f"{network_condition_kwargs['feats']['record'][0].id}.npz")
        
        potentials = get_potentials(steering_args=steering_args, boltz2=True) #(structure=structure, steering_args=steering_args)

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if steering_args["guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        if steering_args.get("use_confidence", False):
            assert (
                confidence_kwargs is not None
            ), "Confidence guidance requires a confidence kwargs"
            assert steering_args["confidence_energy_type"] in ["iptm", "iptm_energy", "ipae", "iplddt"], "Invalid confidence energy type"
            assert steering_args["confidence_guidance_type"] in ["freedom", "ugd"], "Invalid confidence guidance type"
        if confidence_kwargs is not None:
            confidence_module = confidence_kwargs["confidence_module"]
            pred_distogram_logits = confidence_kwargs["pred_distogram_logits"]
            run_confidence_sequentially = confidence_kwargs["run_sequentially"]
            boltz1_confidence_module = confidence_kwargs.get("boltz1_confidence_module", None)
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        token_repr = None
        atom_coords_denoised = None

        # ll for compositional generation
        if isinstance(network_condition_kwargs["s_trunk"], list):
            ll = torch.zeros(len(network_condition_kwargs["s_trunk"]), shape[0], device=self.device)
            vel = torch.zeros(len(network_condition_kwargs["s_trunk"]), *shape, device=self.device)
        else:
            ll = torch.zeros(1, atom_coords.shape[0], device=self.device)
            vel = torch.zeros(1, *shape, device=self.device)

        # logmd for visualization
        if logmd_args["logmd"]:
            from logmd import LogMD
            print("LogMd record id", network_condition_kwargs['feats']['record'][0].id)
            logmd = LogMD() # project=network_condition_kwargs['feats']['record'][0].id
            structure = StructureV2.load(logmd_args["targets_dir"] / f"{network_condition_kwargs['feats']['record'][0].id}.npz")
            ref_coords = None
        elif logmd_args["save_intermediate_predictions"]:
            structure = StructureV2.load(logmd_args["targets_dir"] / f"{network_condition_kwargs['feats']['record'][0].id}.npz")
            ref_coords = None

        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            if verbose:
                print("step_idx", step_idx)
            if logmd_args["logmd"]: # no random augmentation for logmd
                random_R, random_tr = torch.eye(3).to(atom_coords.device).expand(multiplicity, 3, 3), torch.zeros(3, device=atom_coords.device).expand(multiplicity, 1, 3)
            else:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if steering_args["guidance_update"] and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                token_a = torch.zeros(
                    (
                        multiplicity,
                        network_condition_kwargs["feats"]["token_pad_mask"].shape[1],
                        2 * self.token_s,
                    ),
                    device=atom_coords_noisy.device,
                )
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity // max_parallel_samples + 1
                )

                # TODO: implement this for FreeDoM guidance
                #  Check whether to calculate confidence energy and gradient
                # confidence_energy = None
                # calc_confidence_energy = False
                # confidence_grad = None
                # calc_confidence_grad = False
                # if (
                #     steering_args["fk_steering"] 
                #     and ((step_idx % steering_args["fk_resampling_interval"] == 0 and noise_var > 0) or noise_var > 0)
                #     and steering_args.get("use_confidence", False)
                # ):
                #     calc_confidence_energy = True
                # if (
                #     steering_args["guidance_update"]
                #     and step_idx < num_sampling_steps - 1
                #     and steering_args.get("use_confidence", False)
                #     and steering_args["confidence_guidance_interval"] > 0
                #     and step_idx % steering_args["confidence_guidance_every_n_steps"] == 0
                #     and steering_args["confidence_guidance_weight"] > 0
                # ):
                #     calc_confidence_energy = True
                #     calc_confidence_grad = True

                for sample_ids_chunk in sample_ids_chunks:
                    # TODO: implement this for FreeDoM guidance
                    enable_grad = False # calc_confidence_grad and steering_args["confidence_guidance_type"].lower() == "freedom"

                    network_condition_kwargs_chunk = dict(
                        multiplicity=sample_ids_chunk.numel(),
                        **network_condition_kwargs,
                    )
                    ll_chunk = ll[:, sample_ids_chunk]
                    if not superposition:
                        indices = [int(i) for i in sample_ids_chunk.tolist()]
                        network_condition_kwargs_chunk["s_trunk"] = [
                            network_condition_kwargs["s_trunk"][i % len(network_condition_kwargs["s_trunk"])]
                            for i in indices
                        ]
                        network_condition_kwargs_chunk["diffusion_conditioning"] = [
                            network_condition_kwargs["diffusion_conditioning"][i % len(network_condition_kwargs["diffusion_conditioning"])]
                            for i in indices
                        ]
                        ll_chunk = torch.diag(ll[sample_ids_chunk % len(network_condition_kwargs["diffusion_conditioning"]), sample_ids_chunk])
                    
                    with torch.set_grad_enabled(enable_grad):
                        atom_coords_denoised_chunk, token_a_chunk, ll_chunk, vel_chunk = self.preconditioned_network_forward(
                            atom_coords_noisy[sample_ids_chunk],
                            t_hat,
                            network_condition_kwargs=network_condition_kwargs_chunk,
                            step_scale=step_scale,
                            sigma_next=sigma_t,
                            ll=ll_chunk,
                            superposition=superposition,
                            verbose=verbose,
                        )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk
                    if superposition:
                        ll[:, sample_ids_chunk] = ll_chunk
                        vel[:, sample_ids_chunk] = vel_chunk
                    token_a[sample_ids_chunk] = token_a_chunk

                    # TODO: calculate confidence energy and gradient at once for efficiency
                    # if calc_confidence_energy:
                    #     enable_grad = calc_confidence_grad

                    #     sampled_condition_ids = random.sample(range(len(confidence_kwargs["z_trunk"])), k=steering_args["confidence_s_z_samples"])
                    #     confidence_energy_list = []

                    #     for sampled_condition_id in sampled_condition_ids:
                    #         with torch.set_grad_enabled(enable_grad):
                    #             confidence_out = confidence_module(
                    #                 s_inputs=network_condition_kwargs["s_inputs"],
                    #                 s=torch.cat([network_condition_kwargs["s_trunk"][i] for i in sampled_condition_id], dim=0),
                    #                 z=torch.cat([confidence_kwargs["z_trunk"][i] for i in sampled_condition_id], dim=0),
                    #                 x_pred=atom_coords_denoised_chunk,
                    #                 feats=network_condition_kwargs["feats"],
                    #                 pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id],
                    #                 multiplicity=multiplicity,
                    #                 run_sequentially=run_confidence_sequentially,
                    #             )
                    #             if steering_args["confidence_energy_type"] == "iptm":
                    #                 confidence_energy = - confidence_out["iptm"]
                    #             elif steering_args["confidence_energy_type"] == "iptm_energy":
                    #                 confidence_energy = confidence_out["iptm_energy"]
                    #             else:
                    #                 raise ValueError(f"Invalid confidence energy type: {steering_args['confidence_energy_type']}")
                    #             confidence_energy_list.append(confidence_energy)


                if verbose:
                    print("ll", ll)

                iptm_list = None
                boltz1_iptm_list = None
                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)                            
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    if (
                        steering_args.get("use_confidence", False)
                        and (
                            (step_idx >= steering_args["confidence_steering_start"] and step_idx <= steering_args["confidence_steering_end"])
                            or (step_idx == num_sampling_steps - 1)
                        )
                    ):
                        scale_factor = min((1 + steering_args["confidence_tempering_gamma"]) ** (step_idx + 1) - 1, 1.0)
                        inv_temp = scale_factor * steering_args["confidence_resampling_weight"]
                        
                        assert isinstance(confidence_kwargs["z_trunk"], list) and isinstance(network_condition_kwargs["s_trunk"], list) and len(network_condition_kwargs["s_trunk"]) == len(confidence_kwargs["z_trunk"])
                        assert pred_distogram_logits.shape[3] == len(confidence_kwargs["z_trunk"])
                        
                        # sampled_condition_ids = torch.topk(ll, steering_args["confidence_s_z_samples"], dim=0)[1] # shape (multiplicity, top_k)
                        sampled_condition_ids = random.sample(range(len(confidence_kwargs["z_trunk"])), k=steering_args["confidence_s_z_samples"])
                        
                        confidence_energy_list = []
                        iptm_list = []
                        iptm_energy_list = []
                        pae_logits_list = []
                        ipae_list = []
                        iplddt_list = []
                        
                        for sampled_condition_id in sampled_condition_ids: 
                            if verbose:
                                print("s_trunk", network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1).shape)
                                print("z_trunk", confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1).shape)
                                print("atom_coords_denoised", atom_coords_denoised.shape)
                            confidence_out = confidence_module(
                                s_inputs=network_condition_kwargs["s_inputs"],
                                s=network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1),
                                z=confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1),
                                x_pred=atom_coords_denoised,
                                feats=network_condition_kwargs["feats"],
                                pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                multiplicity=multiplicity,
                                run_sequentially=run_confidence_sequentially,
                            )
                            if steering_args["confidence_energy_type"] == "iptm":
                                confidence_energy = - confidence_out["iptm"]
                            elif steering_args["confidence_energy_type"] == "iptm_energy":
                                confidence_energy = confidence_out["iptm_energy"]
                            elif steering_args["confidence_energy_type"] == "ipae":
                                confidence_energy = confidence_out["complex_ipae"]
                            elif steering_args["confidence_energy_type"] == "iplddt":
                                confidence_energy = - confidence_out["complex_iplddt"]
                            else:
                                raise ValueError(f"Invalid confidence energy type: {steering_args['confidence_energy_type']}")

                            iptm_list.append(-confidence_out["iptm"])
                            iptm_energy_list.append(confidence_out["iptm_energy"])
                            pae_logits_list.append(confidence_out["pae_logits"].cpu())
                            ipae_list.append(confidence_out["complex_ipae"].cpu())
                            iplddt_list.append(confidence_out["complex_iplddt"].cpu())

                            print("iptm", confidence_out["iptm"])

                            del confidence_out
                            torch.cuda.empty_cache()

                            if steering_args.get("use_boltz1_confidence_steering", False):
                                boltz1_iptm_list = []
                                boltz1_iptm_energy_list = []
                                boltz1_ipae_list = []
                                boltz1_iplddt_list = []
                                
                                # Use Boltz1 trunk features if provided, otherwise fall back to Boltz2 trunk features
                                if "boltz1_trunk_features" in confidence_kwargs and confidence_kwargs["boltz1_trunk_features"] is not None:
                                    b1_feats = confidence_kwargs["boltz1_trunk_features"]
                                    s_inputs = b1_feats["s_inputs"]
                                    s = b1_feats["s"].expand(multiplicity, -1, -1)
                                    z = b1_feats["z"].expand(multiplicity, -1, -1, -1)
                                else:
                                    s_inputs = network_condition_kwargs["s_inputs"]
                                    s = network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1)
                                    z = confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1)
                                boltz1_confidence_out = boltz1_confidence_module(
                                    s_inputs=s_inputs,
                                    s=s,
                                    z=z,
                                    s_diffusion=None, # token_repr.detach() if token_repr is not None else token_a.detach(),
                                    x_pred=atom_coords_denoised,
                                    feats=network_condition_kwargs["feats"]["boltz1_feats"],
                                    pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                    multiplicity=multiplicity,
                                    run_sequentially=run_confidence_sequentially,
                                )
                                if steering_args["confidence_energy_type"] == "iptm":
                                    boltz1_confidence_energy = - boltz1_confidence_out["iptm"]
                                elif steering_args["confidence_energy_type"] == "iptm_energy":
                                    boltz1_confidence_energy = boltz1_confidence_out["iptm_energy"]
                                elif steering_args["confidence_energy_type"] == "ipae":
                                    boltz1_confidence_energy = boltz1_confidence_out["complex_ipae"]
                                elif steering_args["confidence_energy_type"] == "iplddt":
                                    boltz1_confidence_energy = - boltz1_confidence_out["complex_iplddt"]

                                boltz1_iptm_list.append(-boltz1_confidence_out["iptm"])
                                boltz1_iptm_energy_list.append(boltz1_confidence_out["iptm_energy"])
                                boltz1_ipae_list.append(boltz1_confidence_out["complex_ipae"].cpu())
                                boltz1_iplddt_list.append(boltz1_confidence_out["complex_iplddt"].cpu())

                                print("boltz1_iptm", boltz1_confidence_out["iptm"])

                                del boltz1_confidence_out
                                torch.cuda.empty_cache()

                                confidence_energy = (confidence_energy + boltz1_confidence_energy) / 2

                                print("confidence_energy", confidence_energy)

                            confidence_energy_list.append(confidence_energy)

                        if steering_args["confidence_merge_method"] == "logsumexp":
                            confidence_energy = - torch.logsumexp(-inv_temp * torch.stack(confidence_energy_list, dim=0), dim=0) / inv_temp # 'soft max'
                        elif steering_args["confidence_merge_method"] == "mean":
                            confidence_energy = torch.stack(confidence_energy_list, dim=0).mean(dim=0)
                        elif steering_args["confidence_merge_method"] == "worst":
                            confidence_energy = torch.stack(confidence_energy_list, dim=0).max(dim=0).values
                        elif steering_args["confidence_merge_method"] == "mean_pae":
                            pae_logits = torch.stack(pae_logits_list, dim=0).mean(dim=0).to(atom_coords_denoised.device)
                            _, iptm, _, _, _, _, iptm_energy, _, ipae = compute_ptms(
                                pae_logits, atom_coords_denoised, network_condition_kwargs["feats"], multiplicity
                            )
                            del pae_logits
                            torch.cuda.empty_cache()

                            if steering_args["confidence_energy_type"] == "iptm":
                                confidence_energy = - iptm
                            elif steering_args["confidence_energy_type"] == "iptm_energy":
                                confidence_energy = iptm_energy
                            elif steering_args["confidence_energy_type"] == "ipae":
                                confidence_energy = ipae
                            else:
                                raise ValueError(f"Invalid confidence energy type: {steering_args['confidence_energy_type']}")
                        else:
                            raise ValueError(f"Invalid confidence merge method: {steering_args['confidence_merge_method']}")

                        energy += inv_temp * confidence_energy
                        # if verbose:
                        print("confidence_energy", confidence_energy)
                    rlat = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if energy_traj.shape[1] <= 1:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if steering_args["guidance_update"] and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                    print("log_G", log_G)
                    print("ll_difference", ll_difference)
                    print("resample_weights", resample_weights)

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["guidance_update"]
                    and step_idx < num_sampling_steps - 1
                ):
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        if (
                            steering_args.get("use_confidence", False)
                            and steering_args["confidence_guidance_interval"] > 0
                            and (guidance_step) % steering_args["confidence_guidance_interval"] == 0
                            and step_idx % steering_args["confidence_guidance_every_n_steps"] == 0
                            and steering_args["confidence_guidance_weight"] > 0
                            and step_idx >= steering_args["confidence_steering_start"]
                            and step_idx <= steering_args["confidence_steering_end"]
                        ):
                            scale_factor = min((1 + steering_args["confidence_tempering_gamma"]) ** (step_idx + 1) - 1, 1.0)
                            inv_temp = scale_factor * steering_args["confidence_guidance_weight"]
                            if verbose:
                                print("scale_factor", scale_factor)

                            confidence_grad = torch.empty_like(atom_coords_denoised)

                            torch.autograd.set_detect_anomaly(True)
                            with torch.enable_grad():

                                for idx in range(multiplicity):
                                    coords_for_grad = (
                                        (atom_coords_denoised[idx : idx + 1] + guidance_update[idx : idx + 1])
                                        .clone()
                                        .requires_grad_(True)
                                    )

                                    if isinstance(confidence_kwargs["z_trunk"], list):
                                        assert isinstance(network_condition_kwargs["s_trunk"], list) and len(network_condition_kwargs["s_trunk"]) == len(confidence_kwargs["z_trunk"])
                                        assert pred_distogram_logits.shape[3] == len(confidence_kwargs["z_trunk"])

                                        confidence_energy_list = []
                                        confidence_grad_list = []

                                        # sampled_condition_ids = torch.topk(ll[:, idx], steering_args["confidence_s_z_samples"])[1]
                                        sampled_condition_ids = random.sample(range(len(confidence_kwargs["z_trunk"])), k=steering_args["confidence_s_z_samples"])

                                        for i in sampled_condition_ids:
                                            confidence_out_grad = confidence_module(
                                                s_inputs=network_condition_kwargs["s_inputs"],
                                                s=network_condition_kwargs["s_trunk"][i],
                                                z=confidence_kwargs["z_trunk"][i],
                                                x_pred=coords_for_grad,
                                                feats=network_condition_kwargs["feats"],
                                                pred_distogram_logits=pred_distogram_logits[:, :, :, i],
                                                multiplicity=1,
                                                run_sequentially=run_confidence_sequentially,
                                                differentiable=True,
                                                tau=steering_args["structure_distogram_tau"],
                                            )
                                            if steering_args["confidence_energy_type"] == "iptm":
                                                confidence_energy_grad = - confidence_out_grad["iptm"]
                                            elif steering_args["confidence_energy_type"] == "iptm_energy":
                                                confidence_energy_grad = confidence_out_grad["iptm_energy"]
                                            elif steering_args["confidence_energy_type"] == "ipae":
                                                confidence_energy_grad = confidence_out_grad["complex_ipae"]
                                            elif steering_args["confidence_energy_type"] == "iplddt":
                                                confidence_energy_grad = - confidence_out_grad["complex_iplddt"]
                                            else:
                                                raise ValueError(f"Invalid confidence energy type: {steering_args['confidence_energy_type']}")
                                            
                                            if steering_args.get("use_boltz1_confidence_steering", False):
                                                # Prefer Boltz1 trunk features if provided
                                                if "boltz1_trunk_features" in confidence_kwargs and confidence_kwargs["boltz1_trunk_features"] is not None:
                                                    b1_feats = confidence_kwargs["boltz1_trunk_features"]
                                                    s_inputs = b1_feats["s_inputs"]
                                                    s = b1_feats["s"]
                                                    z = b1_feats["z"]
                                                else:
                                                    s_inputs = network_condition_kwargs["s_inputs"]
                                                    s = network_condition_kwargs["s_trunk"][i]
                                                    z = confidence_kwargs["z_trunk"][i]
                                                boltz1_confidence_out_grad = boltz1_confidence_module(
                                                    s_inputs=s_inputs,
                                                    s=s,
                                                    z=z,
                                                    s_diffusion=None, # token_repr.detach() if token_repr is not None else token_a.detach(),
                                                    x_pred=coords_for_grad,
                                                    feats=network_condition_kwargs["feats"]["boltz1_feats"],
                                                    pred_distogram_logits=pred_distogram_logits[:, :, :, i],
                                                    multiplicity=1,
                                                    run_sequentially=run_confidence_sequentially,
                                                    differentiable=True,
                                                    tau=steering_args["structure_distogram_tau"],
                                                )
                                                if steering_args["confidence_energy_type"] == "iptm":
                                                    boltz1_confidence_energy_grad = - boltz1_confidence_out_grad["iptm"]
                                                elif steering_args["confidence_energy_type"] == "iptm_energy":
                                                    boltz1_confidence_energy_grad = boltz1_confidence_out_grad["iptm_energy"]
                                                elif steering_args["confidence_energy_type"] == "ipae":
                                                    boltz1_confidence_energy_grad = boltz1_confidence_out_grad["complex_ipae"]
                                                elif steering_args["confidence_energy_type"] == "iplddt":
                                                    boltz1_confidence_energy_grad = - boltz1_confidence_out_grad["complex_iplddt"]
                                                confidence_energy_grad = (confidence_energy_grad + boltz1_confidence_energy_grad) / 2

                                            confidence_energy_grad.sum().backward()
                                            confidence_grad_list.append(coords_for_grad.grad.detach().clone())
                                            confidence_energy_list.append(confidence_energy_grad.detach().clone())

                                            coords_for_grad.grad.zero_()

                                        # Gradient of logsumexp
                                        confidence_energy = rearrange(torch.stack(confidence_energy_list, dim=0), "c b -> c b 1 1")
                                        confidence_grad[idx : idx + 1] = \
                                            torch.sum(torch.exp(-inv_temp * confidence_energy) * torch.stack(confidence_grad_list, dim=0), dim=0) \
                                            / torch.sum(torch.exp(-inv_temp * confidence_energy), dim=0)

                                        if verbose:
                                            print("confidence grad norm components", (torch.stack(confidence_grad_list, dim=0)**2).mean((1, 2, 3)).sqrt())
                                            print("confidence grad norm", (confidence_grad[idx : idx + 1]**2).mean((1, 2)).sqrt())

                                    else:
                                        confidence_out_grad = confidence_module(
                                            s_inputs=network_condition_kwargs["s_inputs"],
                                            s=network_condition_kwargs["s_trunk"],
                                            z=confidence_kwargs["z_trunk"],
                                            x_pred=coords_for_grad,
                                            feats=network_condition_kwargs["feats"],
                                            pred_distogram_logits=pred_distogram_logits,
                                            multiplicity=1,
                                            run_sequentially=run_confidence_sequentially,
                                            differentiable=True,
                                            tau=steering_args["structure_distogram_tau"],
                                        )
                                        # confidence_score_grad = (
                                        #     0.8 * confidence_out_grad["iptm"]
                                        #     + 0.2 * confidence_out_grad["ptm"]
                                        # )
                                        if steering_args["confidence_energy_type"] == "iptm":
                                            confidence_energy_grad = - confidence_out_grad["iptm"]
                                        elif steering_args["confidence_energy_type"] == "iptm_energy":
                                            confidence_energy_grad = confidence_out_grad["iptm_energy"]
                                        elif steering_args["confidence_energy_type"] == "ipae":
                                            confidence_energy_grad = confidence_out_grad["complex_ipae"]
                                        elif steering_args["confidence_energy_type"] == "iplddt":
                                            confidence_energy_grad = - confidence_out_grad["complex_iplddt"]
                                        else:
                                            raise ValueError(f"Invalid confidence energy type: {steering_args['confidence_energy_type']}")
                                        confidence_energy_grad.sum().backward()
                                        confidence_grad[
                                            idx : idx + 1
                                        ] = coords_for_grad.grad.detach()

                                energy_gradient += inv_temp * confidence_grad
                                if verbose:
                                    print("atom_coords_denoised norm", (atom_coords_denoised**2).mean().sqrt().item())
                                    print("confidence_grad norm", inv_temp * (confidence_grad**2).mean().sqrt().item())

                            if logmd_args["logmd"]:
                                if ref_coords is None:
                                    ref_coords = atom_coords_denoised[0][atom_mask[0].bool()].clone()
                                    structure.atoms["coords"] = ref_coords.cpu().numpy()
                                else:
                                    with torch.autocast("cuda", enabled=False):
                                        denoised_copy = atom_coords_denoised[0][atom_mask[0].bool()].clone()
                                        structure.atoms["coords"] = weighted_rigid_align(
                                            denoised_copy.reshape(1, -1, 3),
                                            ref_coords.reshape(1, -1, 3),
                                            atom_mask[0][atom_mask[0].bool()].float().reshape(1, -1),
                                            atom_mask[0][atom_mask[0].bool()].float().reshape(1, -1),
                                        ).cpu().numpy() # align to previous structure
                                if logmd_args["logmd_confidence"]:
                                    pdb_str = to_pdb(structure, plddts=confidence_out_grad['plddt'][0], boltz2=True)
                                else:
                                    pdb_str = to_pdb(structure, boltz2=True)
                                pdb_str = "\n".join([line for line in pdb_str.split("\n") if line.startswith("ATOM") or line.startswith("HETATM")])
                                logmd(pdb_str)

                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    ) # dx
                    if verbose:
                        print("guidance dx norm", (scaled_guidance_update**2).mean().sqrt().item())
                    if superposition:
                        ll = ll - (step_scale * vel * (scaled_guidance_update.unsqueeze(0)) / t_hat).sum((-2, -1)) # compensate for the additional guidance update (which effects only the drift)

                # Save intermediate predictions if needed before resampling
                if logmd_args["save_intermediate_predictions"] and (step_idx % logmd_args["save_every_n_steps"] == 0 or step_idx == num_sampling_steps - 1):
                    if ref_coords is None:
                        ref_coords = atom_coords_denoised[0][atom_mask[0].bool()].clone()
                        structure.atoms["coords"] = ref_coords.cpu().numpy()

                    output_dir = logmd_args["prediction_dir"] / network_condition_kwargs["feats"]["record"][0].id / f"step{step_idx}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    for sample_idx, init_atom_coords in enumerate(atom_coords_denoised):
                        structure.atoms["coords"] = weighted_rigid_align(
                            init_atom_coords[atom_mask[0].bool()].reshape(1, -1, 3),
                            ref_coords.reshape(1, -1, 3),
                            atom_mask[0][atom_mask[0].bool()].float().reshape(1, -1),
                            atom_mask[0][atom_mask[0].bool()].float().reshape(1, -1),
                        ).cpu().numpy()
                        pdb_str = to_pdb(structure, boltz2=True)

                        with open(output_dir / f"atom_coords_{sample_idx}.pdb", "w") as f:
                            f.write(pdb_str)

                if logmd_args["save_intermediate_confidence"] and (step_idx % logmd_args["save_every_n_steps"] == 0 or step_idx == num_sampling_steps - 1):
                    inv_temp = steering_args["confidence_resampling_weight"]
                    sampled_condition_ids = list(range(len(confidence_kwargs["z_trunk"])))

                    if iptm_list is None:
                        iptm_list = []
                        iptm_energy_list = []
                        pae_logits_list = []
                        ipae_list = []
                        ipae_std_list = []
                        ipae_entropy_list = []
                        iplddt_list = []
                        for sampled_condition_id in sampled_condition_ids: 
                            for _ in range(steering_args["dropout_samples"]):
                                confidence_out = confidence_module(
                                    s_inputs=network_condition_kwargs["s_inputs"],
                                    s=network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1),
                                    z=confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1),
                                    x_pred=atom_coords_denoised,
                                    feats=network_condition_kwargs["feats"],
                                    pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                    multiplicity=multiplicity,
                                    run_sequentially=run_confidence_sequentially,
                                    use_dropout=steering_args["use_dropout"],
                                )
                                iptm = - confidence_out["iptm"]
                                iptm_energy = confidence_out["iptm_energy"]
                                pae_logits = confidence_out["pae_logits"].cpu()
                                ipae = confidence_out["complex_ipae"].cpu()
                                ipae_std = confidence_out["complex_ipae_std"].cpu()
                                ipae_entropy = confidence_out["complex_ipae_entropy"].cpu()
                                iplddt = confidence_out["complex_iplddt"].cpu()
                                iptm_list.append(iptm)
                                iptm_energy_list.append(iptm_energy)
                                pae_logits_list.append(pae_logits)
                                ipae_list.append(ipae)
                                ipae_std_list.append(ipae_std)
                                ipae_entropy_list.append(ipae_entropy)
                                iplddt_list.append(iplddt)
                            print(f"iptm_list (dropout={steering_args['use_dropout']})", iptm_list)

                        del confidence_out
                        torch.cuda.empty_cache()

                    # Perturbed confidence calculation
                    if logmd_args["save_perturbed_confidence"]:
                        atom_coords_denoised_perturbed = atom_coords_denoised + torch.randn_like(atom_coords_denoised) * logmd_args["confidence_perturbation_scale"] # * self.c_out(torch.tensor(t_hat).to(self.device))

                        iptm_list_perturbed = []
                        iptm_energy_list_perturbed = []
                        ipae_list_perturbed = []
                        iplddt_list_perturbed = []
                        for sampled_condition_id in sampled_condition_ids:
                            confidence_out_perturbed = confidence_module(
                                s_inputs=network_condition_kwargs["s_inputs"],
                                s=network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1),
                                z=confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1),
                                x_pred=atom_coords_denoised_perturbed,
                                feats=network_condition_kwargs["feats"],
                                pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                multiplicity=multiplicity,
                                run_sequentially=run_confidence_sequentially,
                            )
                            iptm_perturbed = - confidence_out_perturbed["iptm"]
                            iptm_energy_perturbed = confidence_out_perturbed["iptm_energy"]
                            ipae_perturbed = confidence_out_perturbed["complex_ipae"].cpu()
                            iplddt_perturbed = confidence_out_perturbed["complex_iplddt"].cpu()
                            iptm_list_perturbed.append(iptm_perturbed)
                            iptm_energy_list_perturbed.append(iptm_energy_perturbed)
                            ipae_list_perturbed.append(ipae_perturbed)
                            iplddt_list_perturbed.append(iplddt_perturbed)

                    if steering_args.get("use_boltz1_confidence_steering", False):
                        if boltz1_iptm_list is None:
                            boltz1_iptm_list = []
                            boltz1_iptm_energy_list = []
                            boltz1_ipae_list = []
                            boltz1_iplddt_list = []

                            # boltz1_iptm_with_boltz2_feats_list = []

                            for sampled_condition_id in sampled_condition_ids: 
                                # Prefer Boltz1 trunk features if provided
                                if "boltz1_trunk_features" in confidence_kwargs and confidence_kwargs["boltz1_trunk_features"] is not None:
                                    b1_feats = confidence_kwargs["boltz1_trunk_features"]
                                    s_inputs = b1_feats["s_inputs"]
                                    s = b1_feats["s"].expand(multiplicity, -1, -1)
                                    z = b1_feats["z"].expand(multiplicity, -1, -1, -1)
                                else:
                                    s_inputs = network_condition_kwargs["s_inputs"]
                                    s = network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1)
                                    z = confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1)
                                boltz1_confidence_out = boltz1_confidence_module(
                                    s_inputs=s_inputs,
                                    s=s,
                                    z=z,
                                    s_diffusion=None, # token_repr.detach() if token_repr is not None else token_a.detach(),
                                    x_pred=atom_coords_denoised,
                                    feats=network_condition_kwargs["feats"]["boltz1_feats"],
                                    pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                    multiplicity=multiplicity,
                                    run_sequentially=run_confidence_sequentially,
                                )
                                boltz1_iptm_list.append(- boltz1_confidence_out["iptm"])
                                boltz1_iptm_energy_list.append(boltz1_confidence_out["iptm_energy"])
                                boltz1_ipae_list.append(boltz1_confidence_out["complex_ipae"].cpu())
                                boltz1_iplddt_list.append(boltz1_confidence_out["complex_iplddt"].cpu())

                                # boltz1_confidence_with_boltz2_feats = boltz1_confidence_module(
                                #     s_inputs=network_condition_kwargs["s_inputs"],
                                #     s=network_condition_kwargs["s_trunk"][sampled_condition_id].expand(multiplicity, -1, -1),
                                #     z=confidence_kwargs["z_trunk"][sampled_condition_id].expand(multiplicity, -1, -1, -1),
                                #     s_diffusion=None, # token_repr.detach() if token_repr is not None else token_a.detach(),
                                #     x_pred=atom_coords_denoised,
                                #     feats=network_condition_kwargs["feats"]["boltz1_feats"],
                                #     pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                #     multiplicity=multiplicity,
                                #     run_sequentially=run_confidence_sequentially,
                                # )
                                # boltz1_iptm_with_boltz2_feats_list.append(- boltz1_confidence_with_boltz2_feats["iptm"])

                            del boltz1_confidence_out
                            torch.cuda.empty_cache()
                        
                        if logmd_args["save_perturbed_confidence"]:
                            boltz1_iptm_list_perturbed = []
                            boltz1_iptm_energy_list_perturbed = []
                            boltz1_ipae_list_perturbed = []
                            boltz1_iplddt_list_perturbed = []
                            for sampled_condition_id in sampled_condition_ids:
                                boltz1_confidence_out_perturbed = boltz1_confidence_module(
                                    s_inputs=s_inputs,
                                    s=s,
                                    z=z,
                                    s_diffusion=None, # token_repr.detach() if token_repr is not None else token_a.detach(),
                                    x_pred=atom_coords_denoised_perturbed,
                                    feats=network_condition_kwargs["feats"]["boltz1_feats"],
                                    pred_distogram_logits=pred_distogram_logits[:, :, :, sampled_condition_id: sampled_condition_id + 1].expand(-1, -1, -1, multiplicity, -1),
                                    multiplicity=multiplicity,
                                    run_sequentially=run_confidence_sequentially,
                                )
                                boltz1_iptm_list_perturbed.append(- boltz1_confidence_out_perturbed["iptm"])
                                boltz1_iptm_energy_list_perturbed.append(boltz1_confidence_out_perturbed["iptm_energy"])
                                boltz1_ipae_list_perturbed.append(boltz1_confidence_out_perturbed["complex_ipae"].cpu())
                                boltz1_iplddt_list_perturbed.append(boltz1_confidence_out_perturbed["complex_iplddt"].cpu())

                    iptm_logsumexp = - (torch.logsumexp(-inv_temp * torch.stack(iptm_list, dim=0), dim=0) - torch.log(torch.tensor(len(iptm_list)).to(self.device))) / inv_temp
                    iptm_mean = torch.stack(iptm_list, dim=0).mean(dim=0)
                    iptm_worst = torch.stack(iptm_list, dim=0).max(dim=0).values
                    iptm_energy_logsumexp = - (torch.logsumexp(-inv_temp * torch.stack(iptm_energy_list, dim=0), dim=0) - torch.log(torch.tensor(len(iptm_energy_list)).to(self.device))) / inv_temp
                    iptm_energy_mean = torch.stack(iptm_energy_list, dim=0).mean(dim=0)
                    iptm_energy_worst = torch.stack(iptm_energy_list, dim=0).max(dim=0).values

                    pae_logits_mean = torch.stack(pae_logits_list, dim=0).mean(dim=0).to(atom_coords_denoised.device)
                    _, iptm_mean_pae, _, _, _, _, iptm_energy_mean_pae, _, _, _, _, _, _ = compute_ptms(
                        pae_logits_mean, atom_coords_denoised, network_condition_kwargs["feats"], multiplicity
                    )
                    iptm_mean_pae = - iptm_mean_pae

                    del pae_logits_mean
                    torch.cuda.empty_cache()

                    ipae_mean = torch.stack(ipae_list, dim=0).mean(dim=0)
                    ipae_std = torch.stack(ipae_std_list, dim=0).mean(dim=0)
                    ipae_entropy = torch.stack(ipae_entropy_list, dim=0).mean(dim=0)
                    iplddt_mean = torch.stack(iplddt_list, dim=0).mean(dim=0)

                    print("iptm_mean", iptm_mean)

                    if steering_args.get("use_boltz1_confidence_steering", False):
                        boltz1_iptm_mean = torch.stack(boltz1_iptm_list, dim=0).mean(dim=0)
                        boltz1_iptm_energy_mean = torch.stack(boltz1_iptm_energy_list, dim=0).mean(dim=0)
                        boltz1_ipae_mean = torch.stack(boltz1_ipae_list, dim=0).mean(dim=0)
                        boltz1_iplddt_mean = torch.stack(boltz1_iplddt_list, dim=0).mean(dim=0)

                        print("boltz1_iptm_mean", boltz1_iptm_mean)

                        # boltz1_iptm_with_boltz2_feats_mean = torch.stack(boltz1_iptm_with_boltz2_feats_list, dim=0).mean(dim=0)
                        # print("boltz1_iptm_with_boltz2_feats_mean", boltz1_iptm_with_boltz2_feats_mean)

                    # Perturbed confidence calculation
                    if logmd_args["save_perturbed_confidence"]:
                        iptm_mean_perturbed = torch.stack(iptm_list_perturbed, dim=0).mean(dim=0)
                        iptm_energy_mean_perturbed = torch.stack(iptm_energy_list_perturbed, dim=0).mean(dim=0)
                        ipae_mean_perturbed = torch.stack(ipae_list_perturbed, dim=0).mean(dim=0)
                        iplddt_mean_perturbed = torch.stack(iplddt_list_perturbed, dim=0).mean(dim=0)

                        if steering_args.get("use_boltz1_confidence_steering", False):
                            boltz1_iptm_mean_perturbed = torch.stack(boltz1_iptm_list_perturbed, dim=0).mean(dim=0)
                            boltz1_iptm_energy_mean_perturbed = torch.stack(boltz1_iptm_energy_list_perturbed, dim=0).mean(dim=0)
                            boltz1_ipae_mean_perturbed = torch.stack(boltz1_ipae_list_perturbed, dim=0).mean(dim=0)
                            boltz1_iplddt_mean_perturbed = torch.stack(boltz1_iplddt_list_perturbed, dim=0).mean(dim=0)

                    # confidence score for each particle
                    scores_per_particle = {}
                    for i in range(len(iptm_list)):
                        scores_per_particle[f'sample_condition_{i}'] = {
                            "iptm": iptm_list[i].cpu().numpy().tolist(),
                            "iptm_energy": iptm_energy_list[i].cpu().numpy().tolist(),
                            "ipae": ipae_list[i].cpu().numpy().tolist(),
                            "iplddt": iplddt_list[i].cpu().numpy().tolist(),
                        }
                    if logmd_args["save_perturbed_confidence"]:
                        for i in range(len(iptm_list_perturbed)):
                            scores_per_particle[f'sample_condition_{i}'].update({
                                "iptm_perturbed": iptm_list_perturbed[i].cpu().numpy().tolist(),
                                "iptm_energy_perturbed": iptm_energy_list_perturbed[i].cpu().numpy().tolist(),
                                "ipae_perturbed": ipae_list_perturbed[i].cpu().numpy().tolist(),
                                "iplddt_perturbed": iplddt_list_perturbed[i].cpu().numpy().tolist(),
                            })

                    # Save confidence scores for all particles in a single file
                    scores = {
                        "iptm_list": [[float(x) for x in t.cpu().numpy()] for t in iptm_list],
                        "iptm_energy_list": [[float(x) for x in t.cpu().numpy()] for t in iptm_energy_list],
                        "ipae_list": [[float(x) for x in t.cpu().numpy()] for t in ipae_list],
                        "iplddt_list": [[float(x) for x in t.cpu().numpy()] for t in iplddt_list],
                    }

                    if steering_args.get("use_boltz1_confidence_steering", False):
                        scores["boltz1_iptm_list"] = [[float(x) for x in t.cpu().numpy()] for t in boltz1_iptm_list]
                        scores["boltz1_iptm_energy_list"] = [[float(x) for x in t.cpu().numpy()] for t in boltz1_iptm_energy_list]
                        scores["boltz1_ipae_list"] = [[float(x) for x in t.cpu().numpy()] for t in boltz1_ipae_list]
                        scores["boltz1_iplddt_list"] = [[float(x) for x in t.cpu().numpy()] for t in boltz1_iplddt_list]

                    for idx in range(multiplicity):
                        # 1 confidence out
                        iptm_1_sample = iptm_list[idx % len(iptm_list)][idx].item()
                        iptm_energy_1_sample = iptm_energy_list[idx % len(iptm_energy_list)][idx].item()
                        ipae_1_sample = ipae_list[idx % len(ipae_list)][idx].item()
                        iplddt_1_sample = iplddt_list[idx % len(iplddt_list)][idx].item()

                        # 4 confidence out
                        random_indices = random.sample(range(len(iptm_list)), k=min(4, len(iptm_list)))
                        iptm_4_samples = torch.stack([iptm_list[i][idx] for i in random_indices]).mean().item()
                        iptm_energy_4_samples = torch.stack([iptm_energy_list[i][idx] for i in random_indices]).mean().item()
                        ipae_4_samples = torch.stack([ipae_list[i][idx] for i in random_indices]).mean().item()
                        iplddt_4_samples = torch.stack([iplddt_list[i][idx] for i in random_indices]).mean().item()

                        scores[f"particle_{idx}"] = {
                            "iptm_logsumexp": iptm_logsumexp[idx].item(),
                            "iptm_mean": iptm_mean[idx].item(),
                            "iptm_worst": iptm_worst[idx].item(),
                            "iptm_mean_pae": iptm_mean_pae[idx].item(),
                            "iptm_energy_logsumexp": iptm_energy_logsumexp[idx].item(),
                            "iptm_energy_mean": iptm_energy_mean[idx].item(),
                            "iptm_energy_worst": iptm_energy_worst[idx].item(),
                            "iptm_energy_mean_pae": iptm_energy_mean_pae[idx].item(),
                            "ipae_mean": ipae_mean[idx].item(),
                            "ipae_std": ipae_std[idx].item(),
                            "ipae_uncertainty_penalty": ipae_mean[idx].item() + ipae_std[idx].item(),
                            "ipae_entropy": ipae_entropy[idx].item(),
                            "iplddt_mean": iplddt_mean[idx].item(),
                            "iptm_1_sample": iptm_1_sample,
                            "iptm_energy_1_sample": iptm_energy_1_sample,
                            "ipae_1_sample": ipae_1_sample,
                            "iplddt_1_sample": iplddt_1_sample,
                            "iptm_4_samples": iptm_4_samples,
                            "iptm_energy_4_samples": iptm_energy_4_samples,
                            "ipae_4_samples": ipae_4_samples,
                            "iplddt_4_samples": iplddt_4_samples,
                        }

                        if logmd_args["save_perturbed_confidence"]:
                        
                            iptm_1_sample_perturbed = iptm_list_perturbed[idx % len(iptm_list_perturbed)][idx].item()
                            iptm_energy_1_sample_perturbed = iptm_energy_list_perturbed[idx % len(iptm_energy_list_perturbed)][idx].item()
                            ipae_1_sample_perturbed = ipae_list_perturbed[idx % len(ipae_list_perturbed)][idx].item()
                            iplddt_1_sample_perturbed = iplddt_list_perturbed[idx % len(iplddt_list_perturbed)][idx].item()

                            iptm_4_samples_perturbed = torch.stack([iptm_list_perturbed[i][idx] for i in random_indices]).mean().item()
                            iptm_energy_4_samples_perturbed = torch.stack([iptm_energy_list_perturbed[i][idx] for i in random_indices]).mean().item()
                            ipae_4_samples_perturbed = torch.stack([ipae_list_perturbed[i][idx] for i in random_indices]).mean().item()
                            iplddt_4_samples_perturbed = torch.stack([iplddt_list_perturbed[i][idx] for i in random_indices]).mean().item()

                            scores[f"particle_{idx}"].update({
                                "iptm_mean_perturbed": iptm_mean_perturbed[idx].item(),
                                "iptm_energy_mean_perturbed": iptm_energy_mean_perturbed[idx].item(),
                                "ipae_mean_perturbed": ipae_mean_perturbed[idx].item(),
                                "iplddt_mean_perturbed": iplddt_mean_perturbed[idx].item(),
                                "iptm_1_sample_perturbed": iptm_1_sample_perturbed,
                                "iptm_energy_1_sample_perturbed": iptm_energy_1_sample_perturbed,
                                "ipae_1_sample_perturbed": ipae_1_sample_perturbed,
                                "iplddt_1_sample_perturbed": iplddt_1_sample_perturbed,
                                "iptm_4_samples_perturbed": iptm_4_samples_perturbed,
                                "iptm_energy_4_samples_perturbed": iptm_energy_4_samples_perturbed,
                                "ipae_4_samples_perturbed": ipae_4_samples_perturbed,
                                "iplddt_4_samples_perturbed": iplddt_4_samples_perturbed
                            })

                        if steering_args.get("use_boltz1_confidence_steering", False):
                            # 1 confidence out boltz1
                            boltz1_iptm_1_sample = boltz1_iptm_list[idx % len(boltz1_iptm_list)][idx].item()
                            boltz1_iptm_energy_1_sample = boltz1_iptm_energy_list[idx % len(boltz1_iptm_energy_list)][idx].item()
                            boltz1_ipae_1_sample = boltz1_ipae_list[idx % len(boltz1_ipae_list)][idx].item()
                            boltz1_iplddt_1_sample = boltz1_iplddt_list[idx % len(boltz1_iplddt_list)][idx].item()

                            # 4 confidence out boltz1
                            boltz1_iptm_4_samples = torch.stack([boltz1_iptm_list[i][idx] for i in random_indices]).mean().item()
                            boltz1_iptm_energy_4_samples = torch.stack([boltz1_iptm_energy_list[i][idx] for i in random_indices]).mean().item()
                            boltz1_ipae_4_samples = torch.stack([boltz1_ipae_list[i][idx] for i in random_indices]).mean().item()
                            boltz1_iplddt_4_samples = torch.stack([boltz1_iplddt_list[i][idx] for i in random_indices]).mean().item()
                            
                            scores[f"particle_{idx}"].update({
                                "iptm_mean_boltz1": boltz1_iptm_mean[idx].item(),
                                "iptm_energy_mean_boltz1": boltz1_iptm_energy_mean[idx].item(),
                                "ipae_mean_boltz1": boltz1_ipae_mean[idx].item(),
                                "iplddt_mean_boltz1": boltz1_iplddt_mean[idx].item(),
                                "iptm_1_sample_boltz1": boltz1_iptm_1_sample,
                                "iptm_energy_1_sample_boltz1": boltz1_iptm_energy_1_sample,
                                "ipae_1_sample_boltz1": boltz1_ipae_1_sample,
                                "iplddt_1_sample_boltz1": boltz1_iplddt_1_sample,
                                "iptm_4_samples_boltz1": boltz1_iptm_4_samples,
                                "iptm_energy_4_samples_boltz1": boltz1_iptm_energy_4_samples,
                                "ipae_4_samples_boltz1": boltz1_ipae_4_samples,
                                "iplddt_4_samples_boltz1": boltz1_iplddt_4_samples,
                                "iptm_mean_mean": (iptm_mean[idx].item() + boltz1_iptm_mean[idx].item()) / 2,
                                "iptm_energy_mean_mean": (iptm_energy_mean[idx].item() + boltz1_iptm_energy_mean[idx].item()) / 2,
                                "ipae_mean_mean": (ipae_mean[idx].item() + boltz1_ipae_mean[idx].item()) / 2,
                                "iplddt_mean_mean": (iplddt_mean[idx].item() + boltz1_iplddt_mean[idx].item()) / 2,
                                "iptm_1_sample_mean": (iptm_1_sample + boltz1_iptm_1_sample) / 2,
                                "iptm_energy_1_sample_mean": (iptm_energy_1_sample + boltz1_iptm_energy_1_sample) / 2,
                                "ipae_1_sample_mean": (ipae_1_sample + boltz1_ipae_1_sample) / 2,
                                "iplddt_1_sample_mean": (iplddt_1_sample + boltz1_iplddt_1_sample) / 2,
                                "iptm_4_samples_mean": (iptm_4_samples + boltz1_iptm_4_samples) / 2,
                                "iptm_energy_4_samples_mean": (iptm_energy_4_samples + boltz1_iptm_energy_4_samples) / 2,
                                "ipae_4_samples_mean": (ipae_4_samples + boltz1_ipae_4_samples) / 2,
                                "iplddt_4_samples_mean": (iplddt_4_samples + boltz1_iplddt_4_samples) / 2,
                            })

                            if logmd_args["save_perturbed_confidence"]:
                                scores[f"particle_{idx}"].update({
                                    "iptm_mean_perturbed_boltz1": boltz1_iptm_mean_perturbed[idx].item(),
                                    "iptm_energy_mean_perturbed_boltz1": boltz1_iptm_energy_mean_perturbed[idx].item(),
                                    "ipae_mean_perturbed_boltz1": boltz1_ipae_mean_perturbed[idx].item(),
                                    "iplddt_mean_perturbed_boltz1": boltz1_iplddt_mean_perturbed[idx].item(),
                                    "iptm_mean_perturbed_mean": (iptm_mean_perturbed[idx].item() + boltz1_iptm_mean_perturbed[idx].item()) / 2,
                                    "iptm_energy_mean_perturbed_mean": (iptm_energy_mean_perturbed[idx].item() + boltz1_iptm_energy_mean_perturbed[idx].item()) / 2,
                                    "ipae_mean_perturbed_mean": (ipae_mean_perturbed[idx].item() + boltz1_ipae_mean_perturbed[idx].item()) / 2,
                                    "iplddt_mean_perturbed_mean": (iplddt_mean_perturbed[idx].item() + boltz1_iplddt_mean_perturbed[idx].item()) / 2,
                                })
                    
                    scores["scores_per_particle"] = scores_per_particle

                    with open(output_dir / "confidence_scores.json", "w") as f:
                        json.dump(scores, f)
                
                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and step_idx >= steering_args["confidence_steering_start"]
                        and step_idx < steering_args["confidence_steering_end"]
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    print("resample_weights", resample_weights)
                    print("resample_indices", resample_indices)

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if steering_args["guidance_update"]:
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]
                    if token_a is not None:
                        token_a = token_a[resample_indices]
                    ll = ll[:, resample_indices]

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(False):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            # Align noisy coordinates to denoised coordinates (Kabsch interpolation)
            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat # velocity = epsilon
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next # next mean

            if not logmd_args["logmd"] and not logmd_args["save_intermediate_predictions"]: continue
            if step_idx < logmd_args["start"]: continue
            if step_idx % logmd_args["interval"] != 0 and step_idx % logmd_args["save_every_n_steps"] != 0: continue

            if ref_coords is None:
                ref_coords = atom_coords_denoised[0][atom_mask[0].bool()].clone()
                structure.atoms["coords"] = ref_coords.cpu().numpy()
            else:
                with torch.autocast("cuda", enabled=False):
                    denoised_copy = atom_coords_denoised[0][atom_mask[0].bool()].clone()
                    structure.atoms["coords"] = weighted_rigid_align(
                        denoised_copy.reshape(1, -1, 3),
                        ref_coords.reshape(1, -1, 3),
                        atom_mask[0][atom_mask[0].bool()].float().reshape(1, -1),
                        atom_mask[0][atom_mask[0].bool()].float().reshape(1, -1),
                    ).cpu().numpy() # align to previous structure
                
            if logmd_args["logmd_confidence"]: 
                confidences = confidence_module(
                    s_inputs=network_condition_kwargs["s_inputs"],
                    s=network_condition_kwargs["s_trunk"][0],
                    z=confidence_kwargs["z_trunk"][0],
                    x_pred=atom_coords_denoised[:1],
                    feats=network_condition_kwargs["feats"],
                    pred_distogram_logits=pred_distogram_logits[:, :, :, 0],
                    multiplicity=1,
                    run_sequentially=run_confidence_sequentially,
                )
                pdb_str = to_pdb(structure, plddts=confidences['plddt'][0], boltz2=True)
            else: 
                pdb_str = to_pdb(structure, boltz2=True)

            pdb_str = "\n".join([line for line in pdb_str.split("\n") if line.startswith("ATOM") or line.startswith("HETATM")])
            if logmd_args["logmd"]:
                logmd(pdb_str)

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords, _ = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        return {
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            noised_atom_coords = out_dict["noised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
