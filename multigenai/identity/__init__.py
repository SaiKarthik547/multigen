"""Identity package — face embedding extraction and character consistency."""
from multigenai.identity.face_encoder import FaceEncoder
from multigenai.identity.identity_resolver import IdentityResolver
from multigenai.identity.identity_latent_encoder import IdentityLatentEncoder

__all__ = ["FaceEncoder", "IdentityResolver", "IdentityLatentEncoder"]
