"""Download the one official encoder asset needed for FastWAM core parity."""

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="SereneC/wan-series-checkpoint",
    revision="fec1e03",
    local_dir="../fastwam_official_assets",
    allow_patterns=[
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
    ],
)
