//! Inter-intra and wedge masks, AV1 7.11.3.11-14.

use super::blend_tables::*;
use super::syntax::Error;

pub(crate) fn wedge(w: usize, h: usize, index: usize, sign: bool) -> Result<Vec<u8>, Error> {
    if !(8..=32).contains(&w) || !(8..=32).contains(&h) || index >= 16 {
        return Err(Error::Invalid("wedge block size or index"));
    }
    let first = [[2, 4, 4], [3, 4, 4], [4, 4, 4], [5, 4, 4]];
    let middle = if h > w {
        [[0, 4, 2], [0, 4, 4], [0, 4, 6], [1, 4, 4]]
    } else if h < w {
        [[1, 2, 4], [1, 4, 4], [1, 6, 4], [0, 4, 4]]
    } else {
        [[0, 4, 2], [0, 4, 6], [1, 2, 4], [1, 6, 4]]
    };
    let last = [
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ];
    let [dir, ox, oy] = if index < 4 {
        first[index]
    } else if index < 8 {
        middle[index - 4]
    } else {
        last[index - 8]
    };
    let xoff = 32 - ((ox * w) >> 3);
    let yoff = 32 - ((oy * h) >> 3);
    let oblique = |y: usize, x: usize| {
        let shift = 16 - y as isize / 2 - isize::from(y & 1 != 0);
        let at = (x as isize - shift).clamp(0, 63) as usize;
        if y & 1 == 0 {
            WEDGE_MASTER_OBLIQUE_EVEN[at]
        } else {
            WEDGE_MASTER_OBLIQUE_ODD[at]
        }
    };
    let master = |y: usize, x: usize| match dir {
        0 => WEDGE_MASTER_VERTICAL[y],
        1 => WEDGE_MASTER_VERTICAL[x],
        2 => oblique(x, y),
        3 => oblique(y, x),
        4 => 64 - oblique(y, 63 - x),
        _ => 64 - oblique(x, 63 - y),
    };
    let sum = (0..w)
        .map(|x| usize::from(master(yoff, xoff + x)))
        .sum::<usize>()
        + (1..h)
            .map(|y| usize::from(master(yoff + y, xoff)))
            .sum::<usize>();
    let flip = (sum + (w + h - 1) / 2) / (w + h - 1) < 32;
    Ok((0..h * w)
        .map(|i| {
            let m = master(yoff + i / w, xoff + i % w);
            if sign == flip { m } else { 64 - m }
        })
        .collect())
}

pub(crate) fn intra_mask(w: usize, h: usize, mode: usize) -> Vec<u8> {
    let scale = 128 / w.max(h);
    (0..w * h)
        .map(|i| match mode {
            1 => II_WEIGHTS_1D[(i / w) * scale],
            2 => II_WEIGHTS_1D[(i % w) * scale],
            3 => II_WEIGHTS_1D[(i / w).min(i % w) * scale],
            _ => 32,
        })
        .collect()
}

pub(crate) fn subsample(
    mask: &[u8],
    width: usize,
    x: usize,
    y: usize,
    subsampling: [bool; 2],
) -> i32 {
    let [sy, sx] = subsampling.map(usize::from);
    let mut sum = 0;
    for dy in 0..1 << sy {
        for dx in 0..1 << sx {
            sum += i32::from(mask[((y << sy) + dy) * width + (x << sx) + dx]);
        }
    }
    let shift = sx + sy;
    if shift == 0 {
        sum
    } else {
        (sum + (1 << (shift - 1))) >> shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wedge_signs_are_complements_and_normalized() {
        for (w, h) in [
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (8, 32),
            (32, 8),
        ] {
            for index in 0..16 {
                let a = wedge(w, h, index, false).unwrap();
                let b = wedge(w, h, index, true).unwrap();
                assert!(a.iter().zip(&b).all(|(&x, &y)| x <= 64 && x + y == 64));
                let sum = (0..w).map(|x| usize::from(a[x])).sum::<usize>()
                    + (1..h).map(|y| usize::from(a[y * w])).sum::<usize>();
                assert!((sum + (w + h - 1) / 2) / (w + h - 1) >= 32);
            }
        }
    }
}
