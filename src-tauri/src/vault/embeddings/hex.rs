use std::io;

pub fn encode_hex_32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

pub fn decode_hex_32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// Decode a 32-byte digest from `[sha256:]<64-hex-chars>[trailing]`. Strips the
/// optional `sha256:` prefix, trims, and tolerates trailing characters past 64.
pub fn parse_digest_hex(s: &str) -> io::Result<[u8; 32]> {
    let s = s.strip_prefix("sha256:").unwrap_or(s).trim();
    if s.len() < 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("digest hex too short: {} chars", s.len()),
        ));
    }
    decode_hex_32(&s[..64])
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-hex character in digest"))
}

/// Scale a vector so its L2 norm is 1. Cosine similarity reduces to a dot
/// product when both operands are L2-normalized (PRD §7.3).
pub fn l2_normalize(v: &mut [f32]) {
    let sum: f32 = v.iter().map(|x| x * x).sum();
    if sum > 0.0 {
        let inv = 1.0 / sum.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
