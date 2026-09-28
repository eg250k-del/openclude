# How to push this project

**Read this before running `git push`.** Getting it wrong has already cost a
whole CI cycle and nearly shipped a broken image.

## The trap

The checkout is normally on a feature branch, not `main`. So:

```powershell
git push origin main     # pushes local `main`, which has not moved
# -> "Everything up-to-date"      <- a lie
```

`git push -q` then hides the output, and the follow-up command that prints
"pushed <sha>" reads the **local** `HEAD`. So the loop looks like it worked,
and the work is sitting on the machine.

This happened for seven commits. It was caught by comparing
`git ls-remote origin refs/heads/main` against `git rev-parse HEAD`, not by
git and not by GitHub.

## The check

Run this after every push. It is the only thing that proves anything landed:

```powershell
git fetch origin
$local = git rev-parse HEAD
$remote = git ls-remote origin "refs/heads/$((git rev-parse --abbrev-ref HEAD))"
if ($local -notmatch $remote.Split()[0]) { "NOT PUSHED" } else { "confirmed" }
```

Or in one line, and it is the safe default:

```powershell
git push --dry-run -v origin HEAD:$(git rev-parse --abbrev-ref HEAD)
# a real line like "4c5f7dc..449fc43  HEAD -> feat/..." means it will work
```

Always push `HEAD:<branch>`, never `<branch>`.

## Why the CI run is the real check

The workflow only builds `main` and `feat/**`, and only when a watched path
changed. Two consequences:

* An **empty commit does not trigger a build.** `paths:` is evaluated against
  the diff, and an empty commit has none. Use `gh workflow run`.
* A dispatch names the ref's SHA **at dispatch time**. If the branch is behind,
  the build silently produces a stale image and still reports success.

So a green `build-image` is not evidence that the image contains your commit.
Check the SHA:

```powershell
gh run list --workflow build-image.yml --limit 1 --json headSha,conclusion
git rev-parse HEAD
```

They must match.

## Getting a new image out

```powershell
git push -u origin HEAD:feat/<branch>     # or merge to main first
gh workflow run build-image.yml --ref <branch>
```

The build takes about 37 minutes. It is a CUDA image built for linux/amd64, so
it goes through QEMU on the runner.

## Verifying a push actually landed, without trusting the message

```powershell
git ls-remote origin refs/heads/<branch>
git rev-list --left-right --count origin/<branch>...HEAD
# the second number is how many commits you are ahead. It must be 0 to be synced.
```
