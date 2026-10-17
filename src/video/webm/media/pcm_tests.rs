use super::*;
use crate::video::backend::AudioSamples;

fn element(id: &[u8], payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() < 16383);
    let mut bytes = id.to_vec();
    bytes.extend_from_slice(&(0x4000 | payload.len() as u16).to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn fixture(codec: &[u8], bits: Option<u16>, packet: &[u8], laced: bool, padding: i64) -> Vec<u8> {
    let mut audio = element(&[0xb5], &1000f64.to_be_bytes());
    audio.extend(element(&[0x9f], &[2]));
    if let Some(bits) = bits { audio.extend(element(&[0x62,0x64], &bits.to_be_bytes())); }
    let mut track = element(&[0xd7], &[1]);
    track.extend(element(&[0x83], &[2]));
    track.extend(element(&[0x86], codec));
    track.extend(element(&[0x56,0xaa], &1_000_000u64.to_be_bytes()));
    track.extend(element(&[0xe1], &audio));
    let mut block = vec![0x81,0,10,if laced { 4 } else { 0 }];
    if laced { block.push(1); }
    block.extend(packet);
    let mut cluster = element(&[0xe7], &[0]);
    if padding == 0 { cluster.extend(element(&[0xa3], &block)); }
    else {
        let mut group = element(&[0xa1], &block);
        group.extend(element(&[0x75,0xa2], &padding.to_be_bytes()));
        cluster.extend(element(&[0xa0], &group));
    }
    let mut segment = element(&[0x16,0x54,0xae,0x6b], &element(&[0xae], &track));
    segment.extend(element(&[0x1f,0x43,0xb6,0x75], &cluster));
    let mut bytes = element(&[0x1a,0x45,0xdf,0xa3], &element(&[0x42,0x82], b"matroska"));
    bytes.extend(element(&[0x18,0x53,0x80,0x67], &segment));
    bytes
}

fn decode(bytes: &[u8], chunk: usize) -> Vec<MediaSample> {
    let mut decoder = WebmMediaDecoder::new();
    let mut output = Vec::new();
    for bytes in bytes.chunks(chunk) {
        output.extend(decoder.push_media(bytes).unwrap());
        while decoder.has_buffered_samples() { output.extend(decoder.drain_packet().unwrap()); }
    }
    decoder.finish().unwrap();
    output
}

#[test]
fn matroska_pcm_streams_endianness_float_and_bit_depth() {
    for (codec,bits,packet) in [
        (b"A_PCM/INT/LIT".as_slice(),16, vec![0,64,0,192]),
        (b"A_PCM/INT/BIG".as_slice(),16, vec![64,0,192,0]),
        (b"A_PCM/FLOAT/IEEE".as_slice(),32, [0.5f32.to_le_bytes(),(-0.5f32).to_le_bytes()].concat()),
        (b"A_PCM/FLOAT/IEEE".as_slice(),64, [0.5f64.to_le_bytes(),(-0.5f64).to_le_bytes()].concat()),
        (b"A_PCM/INT/LIT".as_slice(),8, vec![192,64]),
    ] {
        let bytes = fixture(codec,Some(bits),&packet,false,0);
        for chunk in [1,2,3,7,64,bytes.len()] {
            assert_eq!(decode(&bytes,chunk), vec![MediaSample::Audio {
                timestamp_ns:9_000_000,
                samples:AudioSamples { sample_rate:1000, channels:2, samples:vec![0.5,-0.5] },
            }], "bits={bits} chunk={chunk}");
        }
    }
}

#[test]
fn matroska_pcm_lacing_padding_and_timestamps() {
    let packet = [0,64,0,192,0,32,0,224];
    for padding in [0,1_000_000,-1_000_000] {
        let bytes = fixture(b"A_PCM/INT/LIT",Some(16),&packet,true,padding);
        let samples = decode(&bytes,1);
        let expected = if padding > 0 { vec![(9_000_000,vec![0.5,-0.5])] }
            else if padding < 0 { vec![(10_000_000,vec![0.25,-0.25])] }
            else { vec![(9_000_000,vec![0.5,-0.5]),(10_000_000,vec![0.25,-0.25])] };
        assert_eq!(samples,expected.into_iter().map(|(timestamp_ns,samples)| MediaSample::Audio {
            timestamp_ns, samples:AudioSamples {sample_rate:1000,channels:2,samples}
        }).collect::<Vec<_>>());
    }
}

#[test]
fn matroska_pcm_invalid_packets_report_audio_failure() {
    for (bits,packet) in [(None,vec![0;4]),(Some(16),vec![0;3]),(Some(12),vec![0;4])] {
        let bytes = fixture(b"A_PCM/INT/LIT",bits,&packet,false,0);
        assert!(matches!(decode(&bytes,1).as_slice(),[MediaSample::AudioError(_)]));
    }
}

