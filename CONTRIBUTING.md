# Contributing to 🦫 BEAVR

First off, thank you for considering contributing to **BEAVR Bench**! We value all contributions, code is not the only way to help. Answering questions, improving documentation, reporting bugs, or simply spreading the word are all immensely valuable to the community.

If BeaVR helped you, consider ⭐️ starring the repo, referencing it in your projects/blogs, or sharing your experience.

Please also be mindful to respect our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Ways to Contribute

You can contribute to BEAVR in many ways:

- Fixing bugs or improving existing modules.
- Implementing new robot adapters or extending simulation support.
- Adding new VR input devices (Quest, Vision Pro, LEAP Motion, etc.).
- Contributing to documentation and tutorials.
- Submitting feature requests and reporting issues.

We encourage you to coordinate with the community on our [Discussions](https://github.com/ARCLab-MIT/beavr-bot/discussions) or by opening an issue.

---

## Submitting Issues or Feature Requests

### Found a Bug?

Please:

- Ensure it hasn’t already been reported (search under *Issues*).
- Provide:
  - OS type and version.
  - Python, PyTorch, and CUDA versions.
  - A **minimal code snippet** (ideally runnable in <30s).
  - Full error traceback.
  - Screenshots or logs if relevant.

### Requesting a Feature?

A good feature request includes:

1. **Motivation**
   - Is it solving a limitation or frustration in BeaVR?
   - Is it something you need for a project? Tell us!
   - Is it something you prototyped and think could benefit others? Even better.
2. A short description (one paragraph).
3. (Optional) Example code showing how the feature would be used.
4. Links to related papers, projects, or prior art.
5. Any extra context (sketches, diagrams, etc.).

---

## Adding New Robots, Policies, or Environments



---

## Development Setup



### 3. Install Development Environment

Set up pre-commit hooks:

```bash
pre-commit install
```

Run Tests:

```bash
pytest
```

## Style and Standards

- Code must pass ruff linting/formatting
- Use pre-commit run -all-files before pushing
- Write clear commit messages ([guide](https://cbea.ms/git-commit/))

---

Pull Requests

Before submittingL

1. Ensure your branch is rebased on `upstream/main`.
2. Verify all tests pass
3. Add or update documentation
4. Add tests for new functionality

Submit via GitHub PR. It's normal for maintainers to request changes.

---

## Checklist Before PR

- Clean, descriptive PR title
- Link to related issue(s)
- Mark as draft if WIP
- All tests pass locally

---

Tests

- Install [git lfs](https://git-lfs.com/) if you haven't:

```bash
git lfs install
git lfs pull
```

- Run:

```bash
python -m pytest -sv ./tests
```

---

Thank you!

Every contribution makes BeaVR stronger and more accessible. Whether you fixed a typo, added a driver for a new robot, or extended simulation support, you’re helping the robotics community.
