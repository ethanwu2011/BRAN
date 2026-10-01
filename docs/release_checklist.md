# Publication release checklist

This checklist distinguishes executable code from permission to distribute research artifacts.

## Code checks

- Verify the unchanged architecture hashes.
- Run the generated-input unit tests and end-to-end demonstration.
- Check the real field schema against code definitions without inspecting participant data.
- Preserve the original training and evaluation methods in the research snapshot.
- Identify missing dependencies rather than substituting a new estimator.
- Run the portable figure renderer with generated inputs.
- Scan staged files and review the exact upload allowlist.
- Check the installed package from outside its source directory.

## Author decisions

- MIT license approved by the owner on 1 October 2026.
- Confirm copyright and author metadata before issuing an archival citation.
- State whether trained parameters can be shared separately and under which conditions.
- Approve the final data and code availability statements.
- Archive the exact manuscript code version with a persistent identifier.

The synthetic demonstration, code snapshot and figure renderer do not establish a fresh reproduction of the clinical results. Reproduction on restricted inputs is a separate activity. No real-data study is rerun as part of this packaging task.
