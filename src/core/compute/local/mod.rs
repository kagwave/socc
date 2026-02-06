pub mod types;
pub mod tables;
pub mod rewrite;
pub mod bitwise;
pub mod rotor;
pub mod packed;

pub use rotor::compose_rotor_reference;

pub use types::{LocalBlade, LocalRotor, LocalSector, LocalState};

pub use tables::{
    local_h_action,
    local_s_action,
    mul_local_blades,
    mul_local_rotors,
};

pub use rewrite::{
    local_blade_at,
    set_local_blade,
    local_sector_at,
    set_local_sector,
    local_rotor_at,
    set_local_rotor,
    rewrite_right,
    rewrite_left,
    rewrite_right_rotor,
    rewrite_left_rotor,
    push_blade_through_right_sector,
    push_blade_through_left_sector,
    push_rotor_through_right_sector,
    push_rotor_through_left_sector,
};

pub use bitwise::{
    parity_u64,
    pure_e1_mask,
    rewrite_bitwise,
    rewrite_right_bitwise,
    rewrite_left_bitwise,
    rewrite_rotor_bitwise,
    rewrite_right_rotor_bitwise,
    rewrite_left_rotor_bitwise,
    push_blade_through_right_sector_bitwise,
    push_blade_through_left_sector_bitwise,
    push_rotor_through_right_sector_bitwise,
    push_rotor_through_left_sector_bitwise,
};

pub use packed::PackedBlockTerm;