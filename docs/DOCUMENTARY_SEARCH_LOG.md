# Documentary search log, 27 registry cases

*Pass 1, 2026-11-25. The registry file was not edited. No case was added or removed because of a FAOSTAT 10% flag. The flag file was not opened for this log.*

## Search frame

The universe is the 27 rows in `ancillary/historical_event_registry.csv`: 23 legacy cases and four extension candidates.

Sources opened in this pass are only those already extracted in the repository:

- `docs/MAJOR_FAMINE_LOSS_SOURCES.md`
- `docs/HISTORICAL_LOSSES_METHOD.md`
- `ancillary/historical_loss_evidence_seed.csv`

EM-DAT, UCDP, the IPPC, and the GIEWS archive were not searched. A catalog sentence that says famine, drought, or war is not a screened crop source. Fifteen cases remain `not_yet_screened` for that reason. Absence of a row in the opened extracts is not evidence that no crop report exists.

A disposition means:

- **confirmed**: an opened source reports crop damage, a crop-area change, or a harvest or production shortfall for a stated crop and place inside the case years. The place may be narrower than the registry label. A cause is not assigned.
- **uncertain**: a crop number or crop narrative was opened, and it does not establish the disruption. It may be a production level, a model, a contested harvest, a secondary citation, or an early outlook.
- **contextual_only**: the opened crop or impact statement does not show a crop loss. Food insecurity, mortality, or conflict is not a substitute.
- **not_yet_screened**: no source beyond the registry line was opened.

The machine-readable copy is `ancillary/documentary_search_log_v1.csv`. All 27 identifiers are present once. That file also records evidence availability separately: `not_screened` for the fifteen, `qualitative_only` for NoSummer, and `isolated_numeric` for the other eleven. None has a compatible production time series in this pass.

## Count

| Disposition | Cases |
|---|---:|
| confirmed | 5 |
| uncertain | 6 |
| contextual_only | 1 |
| not_yet_screened | 15 |
| Total | 27 |

## Confirmed

These are crop reports. They are not envelope losses, and they are not attributions.

- **PotatoFamine.** Bourke, via the Ireland CSO, gives potato area planted of 829,875 ha in 1845, 656,014 ha in 1846, and 89,000 ha in 1847. That is a potato-area collapse after blight and seed scarcity. It is not potato tonnes, and potato is outside the current grain envelope. Source: [CSO republication](https://www.cso.ie/en/media/csoie/releasespublications/documents/agriculture/farmingsincethefamine/General_Details_-_Farming_Since_the_Famine.pdf).
- **Drought18761878.** A Tanjore record is reported as putting the total harvest well below normal, and a secondary district-manual note describes very poor Coimbatore crops in 1876–77. The GIPE file did not open when re-retrieved, and the manual page was not opened, so no percent is adopted. The Imperial Gazetteer figure of 83,800 square miles is famine area, not harvested area. The registered label is Asia/Africa. These notes do not cover that label. Source: [Tanjore extract](https://dspace.gipe.ac.in/xmlui/bitstream/handle/10973/36768/GIPE-017895.pdf).
- **ChineseFamine1960.** Lin and Yang, Table 1, report grain output of 200.0 Mt in 1958, 170.0 in 1959, 143.5 in 1960, and 147.5 in 1961. Output in 1960 is 28.25 percent below 1958. This is a national comparison, not a causal estimate, and 1961 is not the onset. Source: [Lin and Yang 2000](https://gwern.net/doc/sociology/2000-lin.pdf).
- **Vanuatu.** The Cyclone Pam assessment reports VT 4,175 million of crop-sector damage and economic losses. The sum mixes food and cash crops and is denominated in money. It confirms crop-sector damage. It is not a cereal tonnage. Source: [post-disaster needs assessment](https://www.dfat.gov.au/sites/default/files/post-disaster-needs-assessment-cyclone-pam.pdf).
- **SudanConflict2023.** The seed extract of the FAO GIEWS brief reports 2025 cereal production of 5.2 Mt, 22 percent below 2024 and 19 percent below the preceding five-year average, for sorghum, millet, and wheat. The decline is not attributed entirely to the conflict. This pass used that summary and did not re-open the brief table. Source: [GIEWS Sudan](https://www.fao.org/giews/countrybrief/country.jsp?code=SDN).

## Uncertain

- **NoSummer.** A UK study describes harvest difficulties in 1816. No cross-region grain total was entered. The registered geography is global. Source: [Veale and Endfield](https://onlinelibrary.wiley.com/doi/10.1111/geoj.12191/full).
- **SovietFamine1921.** Wheatcroft gives three competing USSR-less-SCR evaluations; source block 3a totals 34.11 Mt for 1920 and 27.67 Mt for 1921. The decline depends on the evaluation used, so none is adopted. It is not a Volga loss. Source: [Wheatcroft thesis](https://etheses.bham.ac.uk/id/eprint/530/2/Wheatcroft80PhD2.pdf).
- **DustBowl.** A modeling study estimates a 14 percent mean winter-wheat loss for 1931–1940 against a 1981–2010 climate. Separately, USDA reports 1934 United States wheat production of 526,393 thousand bushels. The first is a decadal model. The second is a national production level, not a Dust Bowl loss. Source: [Science Advances](https://pmc.ncbi.nlm.nih.gov/articles/PMC11290491/).
- **ENSO2015_2016.** A secondary paper cites SADC cereal production 10.2 percent below the previous five-year average. An FAO news note reports a 27 percent maize decline in 2015 relative to 2014 and is an early outlook. The primary SADC report was not opened. Solomon, East Timor, and Laos do not inherit either figure. Source: [NHESS 2020](https://nhess.copernicus.org/articles/20/1187/2020/nhess-20-1187-2020.html).
- **SovietFamine1932.** Secondary discussion reports the 1932 USSR harvest as about 56 million tonnes (Davies and Wheatcroft) and about 50 million tonnes (Tauger). Neither was re-read from its primary page. The estimates disagree, and both are harvest levels rather than one shared baseline. This case is not SovietFamine1921, and Ukraine and Kazakhstan are not separated here. Source: [Davies and Wheatcroft discussion](https://www.tandfonline.com/doi/full/10.1080/09668136.2019.1617464).
- **UkraineWar2022.** A FAO survey of agricultural enterprises reports cultivated area about 7 percent lower in 2024 than in 2023, across 23 oblasts, with inaccessible areas left out ([survey](https://www.fao.org/countryprofiles/news-archive/detail-news/en/c/1742727/)). The GIEWS brief gives 2025 cereal production of 60.8 Mt, about 6 percent below a five-year average and about 8 percent above 2024 ([GIEWS Ukraine](https://www.fao.org/giews/countrybrief/country.jsp?code=UA)). The war is not itself the crop evidence, and the national figure is a level.

## Contextual only

- **SouthSudanConflict.** The 2025 crop and food supply assessment reports traditional smallholder cereal production of 1.2 Mt, 8 percent above 2024 and 27 percent above the five-year average. The crop result in hand is an increase. Persistent food insecurity beside that increase is not a crop loss. Local damage was not extracted. Source: [South Sudan CFSAM release](https://mafs.gov.ss/mafs-and-un-partners-release-cfsam-report-showing-improved-agricultural-production-amid-persistent-food-insecurity-in-south-sudan/).

## Not yet screened

No crop source was opened for these fifteen. Several are famines or wars in the registry. That label is not a disposition.

GreatFamine, Laki1783, SahelDrought2010, MillenniumDrought, NorthKorea1990s, Solomon, EastTimor, Haiti, SierraLeone, Liberia, Yemen, Ethiopia, Laos, Bangladesh, Syria.

Solomon, EastTimor, and Laos stay separate from ENSO2015_2016. Haiti stays one composite case until a crop source is opened for a season and a place. Yemen and Syria stay open-ended. Ethiopia 1983–1985 is this registry row, not the Ethiopia PDR area-code handoff in the boundary audit.

## What this pass does not do

It does not link any case to a FAOSTAT area–crop–year. It does not read the 10% flag list. It does not change `ancillary/historical_event_registry.csv`, the envelopes, or the figures. The next pass is a source search for the fifteen unscreened cases, then a separate link from each case to zero or more production rows.
