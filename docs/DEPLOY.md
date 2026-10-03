# Putting it online

The service runs on a CPU, so it does not need a GPU host. It does need about 2 GB of RAM (torch plus the model), which rules out most free tiers. I have not deployed it anywhere yet, so the steps below are what I would do and not something I have run.

## Before any host

1. Make the repo public, or at least make the release public. The container downloads `best.pt` from a GitHub release on first start, and a private repo returns a 404 to an anonymous download.
2. Upload the current checkpoint as a release. The `v1.0.0` release on this repo is the old checkpoint from before SARD, and it does not match the numbers in the README.

```bash
cp weights/best.pt best.pt
sha256sum best.pt > best.pt.sha256
gh release create v1.1.0 best.pt best.pt.sha256 --title "v1.1.0 SARD-blended checkpoint" \
  --notes "Trained on VisDrone, WiderPerson, AerialPerson and SARD. See ENGINEERING_LOG.md section 22."
```

3. Build the image once locally and look at it: `docker build -t detecsight . && docker run -p 8000:8000 detecsight`, then open `http://localhost:8000`.

The image sets `BATTLESIGHT_PUBLIC_DEMO=1`, which removes `/train` and `/exclude`. Leave that on for anything public. Without it, anybody who finds the URL can start a training run on your host.

## Hugging Face Spaces

This is the one I would pick. A Docker Space is free on CPU (about 16 GB of RAM when I last looked, check the current limits) and gives you a public URL.

1. Create a new Space at huggingface.co/new-space and choose the Docker SDK.
2. A Space reads its settings from the top of its own `README.md`. Do not put these in the GitHub README. Make a branch for it:

```bash
git checkout -b space
```

then add this to the very top of `README.md` on that branch:

```
---
title: DetecSight
sdk: docker
app_port: 8000
---
```

3. Push that branch to the Space:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/detecsight
git push space space:main
```

4. Wait for the build. The first request after it starts downloads the checkpoint, so give it a minute.

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
| `DETECSIGHT_WEIGHTS_TAG` | `v1.1.0` | which release the container downloads |
| `DETECSIGHT_REPO` | `24f2006988/Detecsight` | where it downloads from |

There is no rate limit and no login. If this becomes public and gets used, put something in front of it.
