# Local receipts

Generate a machine-readable receipt after changing or checking out code:

```powershell
.\.venv\Scripts\python.exe scripts\repro_receipt.py `
  --output reproducibility\receipts\local.json `
  --require-artifacts
```

Generated JSON receipts are ignored by Git because they describe one local
worktree and machine. Share one alongside benchmark output when another
engineer needs to audit a particular run. The committed
`reproducibility/artifacts.json` is the stable reference-artifact manifest.
