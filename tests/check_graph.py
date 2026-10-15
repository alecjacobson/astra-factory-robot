"""Audit recursively printed captured graphs, including nested conditionals."""

import sys, json
from simulation.audit import graph_audit

for path in sys.argv[1:]:
    print(path, json.dumps(graph_audit(path)))
