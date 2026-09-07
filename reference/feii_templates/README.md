# Fe II Template Bank

Place optional Fe II templates here as plain text files with wavelength and flux columns.
The loader first checks the user-collected copies in `scripts/fe_template`,
then uses this directory as a working-copy fallback.

Recognized files:

- `scripts/fe_template/fe_uv.txt`: PyQSOFit/Shen UV composite, commonly described as combining Vestergaard & Wilkes (2001), Salviander et al., and Tsuzuki et al. (2006).
- `scripts/fe_template/fe_optical.txt`: PyQSOFit optical Fe II template.
- `reference/feii_templates/fe_uv.txt`: fallback copy of the same UV template.
- `reference/feii_templates/fe_optical.txt`: fallback copy of the same optical template.
- `vestergaard_wilkes_2001.txt`: direct VW01 UV template, if you have the original table.
- `tsuzuki_2006.txt`: direct Tsuzuki et al. UV/optical template, if you have the original table.

The mock generator automatically loads files listed in `scripts/feii_templates.py` when they exist.
