
## Code Structure

- `eif_map.py`: Generates the Expected Information Field (EIF) through `map_generate(random_seed=0)`.
- `traj_generater.py`: Implements trajectory planning and gradient-based trajectory optimization.

## Examples

### Informative Path Planning (EIF + SDF)

```bash
python3 informative_path_planner.py
````

Visualizes informative path planning using the proposed EIF together with the Signed Distance Field (SDF) for obstacle avoidance.

### EIF Generation

```bash
python3 test_eif_generation.py
```

Demonstrates the generation of the Expected Information Field.

```


