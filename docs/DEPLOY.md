# Putting it online

The service runs on a CPU, so it does not need a GPU host. It does need about 2 GB of RAM (torch plus the model), which rules out most free tiers. I have not deployed it anywhere yet, so the steps below are what I would do and not something I have run.

## Before any host

1. Make the repo public, or at least make the release public. The container downloads `best.pt` from a GitHub release on first start, and a private repo returns a 404 to an anonymous download.
2. The `v1.0.0` release carries the checkpoint the README numbers come from (the SARD-blended one). If you retrain, upload the new one as a new release and set `DETECSIGHT_WEIGHTS_TAG` to it.

3. Build the image once locally and look at it: `docker build -t detecsight . && docker run -p 8000:8000 detecsight`, then open `http://localhost:8000`.

The image sets `BATTLESIGHT_PUBLIC_DEMO=1`, which removes `/train` and `/exclude`. Leave that on for anything public. Without it, anybody who finds the URL can start a training run on your host.

## Hugging Face Spaces

This is the one I would pick. A Docker Space is free on CPU (about 16 GB of RAM when I last looked, check the current limits) and gives you a public URL.

1. Create a new Space at huggingface.co/new-space and choose the Docker SDK.
2. Log in on your machine: `hf auth login --add-to-git-credential` (the token needs write access).
3. From this repo, run `bash scripts/deploy_space.sh <your-username>/<space-name>`.

The script pushes a fresh one-commit copy of the repo with the few header lines a Space needs at the top of its README. It does not push this repo's history, because Hugging Face rejects any history that contains a file over 10 MB and the first commit here has an old checkpoint in it. `DRY_RUN=1` builds the copy and stops before the push.

Wait for the build. The first request after it starts downloads the checkpoint, so give it a minute.

Spaces go to sleep when nobody uses them and take a while to wake up, which is fine for a portfolio link.

## Other hosts

| host | works? |
|---|---|
| Render or Railway free tier | no, about 512 MB of RAM, torch alone is more than that |
| Fly.io | yes with a 2 GB machine, but that costs a few dollars a month |
| any VPS with 2 GB or more | yes, `docker run -d -p 80:8000 --restart unless-stopped detecsight` |
| a GPU host | works, but the image is CPU only. You would need a CUDA base image, and nothing free has one |

Render and similar hosts set a `PORT` variable and the image listens on it.

## Settings

| variable | default | what it does |
|---|---|---|
| `BATTLESIGHT_PUBLIC_DEMO` | `1` in the image, `0` otherwise | removes `/train` and `/exclude` |
| `BATTLESIGHT_CORS_ORIGINS` | `*` | comma separated list of allowed origins |
| `BATTLESIGHT_MAX_UPLOAD_MB` | `10` | bigger uploads get a 413 |
| `BATTLESIGHT_IMGSZ` | `1280` | lower it to `960` or `640` if the host is too slow, but small people get missed |
| `DETECSIGHT_WEIGHTS_TAG` | `v1.0.0` | which release the container downloads |
| `DETECSIGHT_REPO` | `24f2006988/Detecsight` | where it downloads from |

There is no rate limit and no login. If this becomes public and gets used, put something in front of it.
