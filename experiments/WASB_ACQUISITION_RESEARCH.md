# WASB: practical public acquisition and terms evidence

Prepared 2026-10-25. Follow-up to
[SOCCER_BALL_TRACKING_RESEARCH.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/SOCCER_BALL_TRACKING_RESEARCH.md).
This does not prejudge the running Round 37 representation experiment. Only
this new note was written. No `data/` paths were read, existing files edited,
weights/videos downloaded, archive label contents extracted, fits/scores run,
packages installed, new terms accepted or external contacts made. Checks used
public source text, HTTP metadata and **1,108 bytes of ZIP directory/header
metadata**, containing no video or annotation payload.

## Actionable result

The official revised labels and camera 5/6 videos are publicly reachable without
an account. There is a bounded route: later acquire the **87,744-byte annotation
ZIP** and only the two test-camera compressed video members, **87,097,245 bytes
combined**, using verified HTTP byte ranges. Downloading the entire
306,355,561-byte six-camera video ZIP is unnecessary. Both files require proper
identity/format checks before use; this research did not acquire their payloads.

The official project clearly intends evaluation of its published pretrained
weights. Its project/code license is MIT, with no separate restrictive binary
terms found. No text expressly naming the Drive checkpoint as MIT was found
either. Record that evidence accurately; do not invent a prohibition, a mandatory
author-contact step or additional acceptance requirement. Conversely, do not
claim that commercial deployment or redistribution rights for the separately
hosted binary were explicitly verified. The original videos have their own
published non-commercial research terms, which remain distinct from code and
model-artifact licensing.

## Terms: what the first-party material says

All repository references below are pinned to
**`923462cacdeb3353b84ddebdedb3f4b7a8553b0f`**.

| First-party evidence | Supported conclusion |
|---|---|
| [README](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/README.md) carries an MIT badge, describes pretrained evaluation code and links the model zoo | MIT is the project's announced license; pretrained evaluation is an advertised use |
| [LICENSE.md](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/LICENSE.md) grants rights in software and associated documentation, retaining NTT copyright/notice | Code and its notices are covered; the text does not specifically enumerate Drive weight binaries |
| [MODEL_ZOO.md](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md) says “We provide model weights” and directly links the Soccer/WASB artifact | This is author-published model distribution, not an unverified third-party mirror |
| [GET_STARTED.md](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/GET_STARTED.md) instructs downloading and evaluating `wasb_soccer_best.pth.tar` | The public research-evaluation route is affirmatively documented; no separate account or agreement step is specified |
| [Original dataset provider](https://pspagnolo.jimdofree.com/download/) permits non-commercial research in video signal processing and requests the D'Orazio et al. 2009 citation | Dataset conditions exist even though the download does not present a click-through agreement |

A current public issue asks about tennis-checkpoint licensing. At inspection its
single reply is another non-author question, not a rights-holder clarification.
It establishes neither a new restriction nor a soccer-model grant.
[Issue and discussion](https://github.com/nttcom/WASB-SBDT/issues/12).
There is no reason to contact anyone as part of this bounded research task.

## Revised ball annotations: metadata verified, payload unopened

The pinned author's
[setup_soccer.sh](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/setup_scripts/setup_soccer.sh)
identifies this revised annotation archive separately from the original dataset's
annotation link:

- [Publisher-linked Drive page](https://drive.google.com/file/d/1Oqr33EXOEjERothXvVAqAeKiKuB-per1/view).
- [Direct public artifact endpoint](https://drive.usercontent.google.com/download?id=1Oqr33EXOEjERothXvVAqAeKiKuB-per1&export=download).
- HEAD: HTTP 200, `application/octet-stream`, filename `soccer_annos.zip`,
  **87,744 bytes**, `Accept-Ranges: bytes`, last-modified 2023-11-16.
- Verified ZIP EOCD: last 22 bytes, no ZIP comment; six entries; central
  directory at byte 87,362 with length 360. Both metadata requests returned
  HTTP 206 with exact requested Content-Range.

The six archive members are `annos/ID-1.xml` through `annos/ID-6.xml`.
Relevant directory entries:

| Member | Compressed bytes | Uncompressed bytes | CRC32 | Local-header offset |
|---|---:|---:|---|---:|
| `annos/ID-5.xml` | 4,641 | 58,156 | `6b4146db` | 79,012 |
| `annos/ID-6.xml` | 3,621 | 44,559 | `79c33891` | 83,697 |

The **expected content schema** is verified from the official
[soccer XML reader](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/datasets/soccer.py):
tracks contain point elements with integer `frame`, `points="x,y"`, `outside`,
`occluded` and an attribute `used_in_game`. The loader uses the in-game ball,
maps `occluded==0` to visibility and fills absent frame entries as invisible.
These are image **center/visibility** labels, not ownership labels or boxes.
Actual XML well-formedness, frame coverage and values remain unverified because
no member was decompressed. A future ingestion audit must check those facts;
the loader's absent→invisible rule is not itself proof of complete annotation.

A complete future label acquisition is only 87.7 kB, simpler than partial-member
retrieval. Verify length, six expected safe paths, each XML's size/CRC, and retain
local SHA256s. No publisher SHA256 was located. Do not silently reuse the original
dataset's older ball labels or apply its old first-300-frame convention without
checking the revised frame keys.

## Video camera 5/6: exact bounded ranges verified

The official setup script and dataset provider link:
[Sequences.zip](https://www.dropbox.com/s/3zhms9iv4gy3k81/Sequences.zip?dl=1).
An unauthenticated HEAD followed Dropbox's public redirect and returned HTTP 200,
**306,355,561 bytes**, attachment `Sequences.zip`, `Accept-Ranges: bytes`.
No cookie jar or authorization header was used. No login, NDA or interactive
acceptance was encountered. This establishes current public reachability, not
permanent future access or the absence of the provider's published use terms.

Metadata-only GET ranges returned HTTP 206 and exact Content-Range:

- EOCD `306355539–306355560`: 22 bytes, zero comment.
- Central directory `306354987–306355538`: 552 bytes, six video members.
- The two local headers below: 76 bytes each, confirming method 8 (DEFLATE),
  flags 0, 46-byte names, zero extra bytes and no encryption/data descriptor.

| Camera | Member name | Local header | **Future compressed payload range, inclusive** |
|---|---|---|---|
| 5 | `Film Role-0 ID-5 T-2 m00s00-000-m00s00-185.avi` | `219257590–219257665` | **`219257666–262709089`** |
| 6 | `Film Role-0 ID-6 T-2 m00s00-000-m00s00-185.avi` | `262709090–262709165` | **`262709166–306354986`** |

| Camera | Compressed payload | Resulting AVI bytes | CRC32 |
|---|---:|---:|---|
| 5 | 43,451,424 | 44,721,664 | `145b43a2` |
| 6 | 43,645,821 | 44,980,736 | `42084099` |

Before a future media request, revalidate archive total length, directory names,
offsets and local headers against these records. If the server ignores Range,
returns a login/error page, changes size or returns HTTP 200 for a range, **stop
before reading the response body**. Do not accidentally fetch the whole archive.
Retrieve the two exact compressed payloads only, stream-decompress raw DEFLATE
(`zlib` window bits -15), enforce the declared output ceilings and verify exact
decompressed length, CRC, stream EOF and SHA256. Save only those two AVI members
under canonical local camera IDs. This is less than 88 MB of media transfer;
it requires neither arbitrary ZIP extraction nor downloading cameras 1–4.

These are compressed ZIP-member byte ranges, **not frame-seek ranges** inside
an AVI. Arbitrary temporal seeking cannot safely be inferred from these offsets.
The bounded actionable route is to obtain each complete camera member, then
locally inspect codec/fps/frame count and decode only a prospectively fixed set
of adjacent three-frame samples. No claim is made that an arbitrary short
compressed prefix will be playable or accurately seekable.

The pinned [dataset split](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/configs/dataset/soccer.yaml)
uses cameras 1–4 for training and 5/6 for testing. The
[paper](https://arxiv.org/html/2311.05237v1) explains that these are synchronized
views of the same play. A 5/6 sample is an adapter/checkpoint reproduction check,
not an unseen-match transfer claim.

## No labeled example images are shipped in the pinned repository

The complete non-truncated
[pinned Git tree](https://api.github.com/repos/nttcom/WASB-SBDT/git/trees/923462cacdeb3353b84ddebdedb3f4b7a8553b0f?recursive=1)
contains source/configuration/setup scripts and documentation, with **no image,
video, ball-center annotation or synthetic example dataset blob**. The README's
external teaser video is a demonstration, not a labeled adapter fixture.
Do not invent ball centers by inspecting paper figures or the teaser.

Source-only synthetic sanity remains useful: construct known-coordinate impulse
images and heatmaps to check RGB/temporal channel ordering, affine round trips,
eight-connected components, weighted centers and inverse mapping. Those tests
verify adapter math; they cannot require a soccer-trained detector to recognize
an artificial dot or establish field accuracy. The first real labeled visual
sanity sample therefore comes from the revised XML plus test-camera video route.

## Bounded next action

After Round 37 evidence is available, preserve its outcome and freeze the next
phase separately. The existing user's research scope supports preparing a local
reproduction of the author's documented evaluation without new contacts or an
invented permission ceremony. Record code notices, the author's weight links,
the absence of separate binary wording and the video's research-use conditions
accurately in that phase's manifest.

Start with the one 6,102,633-byte soccer checkpoint already identified in the
preceding note, pinned source/config and synthetic strict-load/runtime tests.
If those pass the frozen resource budget, acquire the small revised label ZIP
and only cameras 5/6 as above. Validate frame indexing against actual video
metadata before any metric; freeze a small visible/invisible evaluation sample
before detector outputs, with an explicit support failure if insufficient labels
exist. No competition observations, public possession fitting or selection need
enter that technical/external sanity phase. Native opportunity and substantial
possession-improvement gates remain subsequent requirements, unchanged.
