# Generate chagre for non-canonical nucleotides

This folder has two examples:
- `cap` is the 5' cap structure (without the tri-phosphate group)
- `locked_nucleotide` is a locked nucleotide.

### Usage
Run the following command to generate the charge for the nucleotide:
```bash
cd ./cap
python get_cap_charge.py cap_M7G.yaml
```

```bash
cd ./locked_nucleotide
python get_nucleotides_charge_resp2.py nucleotides_LNA.yaml
```

Then to make the Amber compatibale `.lib` files, run:
```bash
cd generate_amber_libs
python make_fragement_<option>.py
```