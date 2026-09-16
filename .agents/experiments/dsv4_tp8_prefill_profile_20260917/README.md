# Literal launch-profile consolidation (pending service replay)

The accepted8K route-producer launcher has166 lines after accumulating
independent experiment overrides. The generated109-line `clean-launcher.sh`
keeps each variable's FINAL literal export/unset and the same working directory,
strict shell options and final command. It does not change global defaults.

`normalize.py` refuses expansions/control flow or code after the final exec.
`--check clean-launcher.sh` replaces only the server exec with `/usr/bin/env -0`,
then compares complete terminal environments from two isolated shell seeds,
including stale inherited overrides and an untouched sentinel. This never
executes the server. `test_normalize.py` also checks last-assignment semantics,
unsets,empty values,quoted spaces,idempotence and rejection of dynamic code.

CPU equality is NOT fresh-process service validation. The active32K ABBA keeps
its existing frozen script. Only after all regressions stop should a separate
service replay validate this equivalent startup path,1Mpool,backend hits,
performance and fixed continuations. The measured original stays unchanged.
