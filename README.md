# `assets` branch

Binary media referenced by documentation on `main`, kept on an orphan branch so
it never lands in a sparse or shallow clone of a recipe.

> **Do not delete this branch.** Nothing in the `main` tree points at it — the
> files are referenced by raw URL only, so deleting the branch silently breaks
> images in published documentation.

| File | Referenced by |
|---|---|
| `long-horizon-harness/horizon-demo.gif` | `core/python/long-horizon-harness/README.md` |

Files are served from:

```
https://raw.githubusercontent.com/google/adk-samples/assets/<path>
```

## Adding a file

Media here should be compressed before committing. For screen-recording GIFs,
dithering is the main thing to avoid — it adds per-pixel noise that defeats
GIF's run-length compression on flat UI surfaces:

```bash
ffmpeg -i in.gif -vf "fps=12,scale=1684:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff" -y pal.png
ffmpeg -i in.gif -i pal.png -lavfi "fps=12,scale=1684:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle" -y out.gif
gifsicle -O3 --lossy=80 out.gif -o final.gif
```

That reduced `horizon-demo.gif` from 11.5 MB to 7.7 MB at unchanged resolution.
