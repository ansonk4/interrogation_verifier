Example question:

Provider A costs 100 dollars for 40 units.
Provider B costs 90 dollars for 30 units.
Which provider has the lower unit price?

Agent answer:

A

Agent reasoning:

A costs 100/40 = 2.5 per unit. 
B costs 90/30 = 3 per unit. 
Since 2.5 < 3, choose A.

Graph:

```json
{
"nodes": [
    {
    "id": "n1",
    "claim": "100 / 40 = 2.5",
    "kind": "calculation",
    "sources": ["reasoning"]
    },
    {
    "id": "n2",
    "claim": "90 / 30 = 3",
    "kind": "calculation",
    "sources": ["reasoning"]
    }
],
"edges": [
    {
    "id": "e1",
    "premise_node_ids": ["n1", "n2"],
    "claim": "2.5 < 3",
    "conclusion": "answer A"
    }
],
"coverage_claim": "n1,n2 -> e1 -> answer A"
}
```