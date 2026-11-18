//! Binary adaptive arithmetic coding (a range coder with carry propagation)
//! and the `Coder` trait that lets one syntax function serve the encoder, the
//! decoder and the rate estimator used by rate-distortion decisions.

/// an adaptive probability: two estimators at different speeds, averaged.
/// `p` is the probability of a 0 in 1/32768ths.
#[derive(Clone, Copy, Debug)]
pub struct Prob {
    fast: u16,
    slow: u16,
}

impl Default for Prob {
    fn default() -> Self {
        Prob { fast: 16384, slow: 16384 }
    }
}

impl Prob {
    #[inline]
    pub fn p0(&self) -> u32 {
        ((self.fast as u32 + self.slow as u32) >> 1).clamp(32, 32768 - 32)
    }
    #[inline]
    pub fn update(&mut self, bit: bool) {
        if bit {
            self.fast -= self.fast >> 4;
            self.slow -= self.slow >> 7;
        } else {
            self.fast += (32768 - self.fast) >> 4;
            self.slow += (32768 - self.slow) >> 7;
        }
    }
}

pub trait Coder {
    /// codes one context-modelled bit; the encoder returns `b`, the decoder the decoded bit
    fn bit(&mut self, p: &mut Prob, b: bool) -> bool;
    /// one equiprobable bit
    fn bypass(&mut self, b: bool) -> bool;
    /// `n` equiprobable bits, most significant first
    fn bypass_bits(&mut self, n: u32, v: u32) -> u32 {
        let mut out = 0;
        for i in (0..n).rev() {
            out = out << 1 | self.bypass((v >> i) & 1 != 0) as u32;
        }
        out
    }
    /// true for the decoder: syntax functions then fill in the values
    fn decoding(&self) -> bool;
}

pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    pub out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Encoder { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1, out: Vec::new() }
    }
}

impl Encoder {
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low as u32) >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = ((self.low as u32) << 8) as u64;
    }
    #[inline]
    fn normalize(&mut self) {
        while self.range < 1 << 24 {
            self.range <<= 8;
            self.shift_low();
        }
    }
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        // the first byte is always 0 (the initial cache): drop it
        self.out.remove(0);
        self.out
    }
    /// bytes so far (an estimate of the final size)
    pub fn len(&self) -> usize {
        self.out.len() + self.cache_size as usize + 4
    }
}

impl Coder for Encoder {
    #[inline]
    fn bit(&mut self, p: &mut Prob, b: bool) -> bool {
        let bound = (self.range >> 15) * p.p0();
        if b {
            self.low += bound as u64;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        p.update(b);
        self.normalize();
        b
    }
    #[inline]
    fn bypass(&mut self, b: bool) -> bool {
        self.range >>= 1;
        if b {
            self.low += self.range as u64;
        }
        self.normalize();
        b
    }
    fn decoding(&self) -> bool {
        false
    }
}

pub struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    code: u32,
    range: u32,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { data, pos: 0, code: 0, range: 0xFFFF_FFFF };
        for _ in 0..4 {
            d.code = d.code << 8 | d.next() as u32;
        }
        d
    }
    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
    #[inline]
    fn normalize(&mut self) {
        while self.range < 1 << 24 {
            self.range <<= 8;
            self.code = self.code << 8 | self.next() as u32;
        }
    }
    /// true once the decoder has read past the end (a truncated or corrupt stream)
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len() + 4
    }
}

impl Coder for Decoder<'_> {
    #[inline]
    fn bit(&mut self, p: &mut Prob, _b: bool) -> bool {
        let bound = (self.range >> 15) * p.p0();
        let b = if self.code < bound {
            self.range = bound;
            false
        } else {
            self.code -= bound;
            self.range -= bound;
            true
        };
        p.update(b);
        self.normalize();
        b
    }
    #[inline]
    fn bypass(&mut self, _b: bool) -> bool {
        self.range >>= 1;
        let b = self.code >= self.range;
        if b {
            self.code -= self.range;
        }
        self.normalize();
        b
    }
    fn decoding(&self) -> bool {
        true
    }
}

/// -log2(p) in 1/256 bit, for p = i/1024
fn cost_table() -> &'static [u16; 1025] {
    static T: std::sync::OnceLock<[u16; 1025]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u16; 1025];
        for (i, v) in t.iter_mut().enumerate() {
            let p = (i.max(1) as f64) / 1024.0;
            *v = (-p.log2() * 256.0).round().min(65535.0) as u16;
        }
        t
    })
}

/// counts bits (in 1/256 bit) without producing output or adapting the models
#[derive(Default)]
pub struct Estimator {
    pub cost: u32,
}

impl Coder for Estimator {
    #[inline]
    fn bit(&mut self, p: &mut Prob, b: bool) -> bool {
        let p0 = p.p0();
        let pb = if b { 32768 - p0 } else { p0 };
        self.cost += cost_table()[(pb >> 5) as usize] as u32;
        b
    }
    #[inline]
    fn bypass(&mut self, b: bool) -> bool {
        self.cost += 256;
        b
    }
    fn decoding(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_with_skewed_and_bypass_bits() {
        let mut s = 99u64;
        let mut bits = vec![];
        for i in 0..20000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let skew = (s >> 33) % 100 < if i % 3 == 0 { 50 } else { 7 };
            bits.push(skew);
        }
        let mut e = Encoder::default();
        let mut ctx = [Prob::default(); 3];
        for (i, &b) in bits.iter().enumerate() {
            if i % 5 == 4 {
                e.bypass(b);
            } else {
                e.bit(&mut ctx[i % 3], b);
            }
        }
        e.bypass_bits(13, 0x1ABC);
        let data = e.finish();
        let mut d = Decoder::new(&data);
        let mut ctx = [Prob::default(); 3];
        for (i, &b) in bits.iter().enumerate() {
            let got = if i % 5 == 4 { d.bypass(false) } else { d.bit(&mut ctx[i % 3], false) };
            assert_eq!(got, b, "bit {i}");
        }
        assert_eq!(d.bypass_bits(13, 0), 0x1ABC);
        // skewed bits compress: well under one bit each
        assert!(data.len() * 8 < bits.len() * 3 / 4, "{} bytes", data.len());
    }

    #[test]
    fn carries_propagate() {
        // long runs of likely-1 bits push `low` toward overflow
        let mut e = Encoder::default();
        let mut p = Prob::default();
        let pattern: Vec<bool> = (0..50000).map(|i| i % 97 != 0).collect();
        for &b in &pattern {
            e.bit(&mut p, b);
        }
        let data = e.finish();
        let mut d = Decoder::new(&data);
        let mut p = Prob::default();
        for (i, &b) in pattern.iter().enumerate() {
            assert_eq!(d.bit(&mut p, false), b, "bit {i}");
        }
    }

    #[test]
    fn the_estimate_tracks_the_real_size() {
        let mut e = Encoder::default();
        let mut est = Estimator::default();
        let mut p = Prob::default();
        let mut q = Prob::default();
        for i in 0..10000 {
            let b = i % 10 == 0;
            est.bit(&mut q, b);
            q.update(b);
            e.bit(&mut p, b);
        }
        let real = e.finish().len() as f64 * 8.0;
        let guess = est.cost as f64 / 256.0;
        assert!((real - guess).abs() / real < 0.03, "real {real} estimate {guess}");
    }
}
