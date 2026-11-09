"""Shared target-free inference entrypoint for the corrected TabPFN hybrid."""
from control_refinement import contact_support
from factorized_latent import decode_owner
from motion_dead_trim import decode_motion_dead_trim
from tabpfn_short_control import restore_supported_islands


def decode_tabpfn_short_control(arrays, owner, supported, broadcast, motion, raw, blocks):
    reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
    alternative = decode_owner(arrays, owner, supported, broadcast, motion, raw, blocks)
    return restore_supported_islands(reference, alternative, contact_support(raw), blocks)
