# Archival reconstruction, pre-1961 cases

*Pass of 2026-11-25. These figures were read from the cited pages. They are not FAOSTAT rows, they are not appended to the QCL panel, and they are not envelope losses. The registry was not edited. The 10% flag file was not opened.*

The separate standard is the one in protocol section 2C. A pre-1961 case may carry archival observations when a bounded source gives a crop, a place, a year, and a unit. A missing year stays missing. A second evaluation of the same harvest stays beside the first. Procurement, trade, mortality, and famine area are not production. Potato stays off the cereal axis. A district figure stays a district figure.

The observation table is `ancillary/archival_reconstruction_v1.csv`. Coverage of the nine registry cases that begin before 1961 is `ancillary/archival_reconstruction_coverage_v1.csv`. Both are written by `scripts/build_archival_reconstruction.py`.

## Count

| Coverage | Cases |
|---|---:|
| Partial extraction from a re-read source | 3 |
| Named collection not re-opened | 2 |
| Outside this frame | 4 |
| Total pre-1961 cases | 9 |

The observation table has 104 rows: 65 for China, 27 for the 1921 Soviet case, and 12 for Ireland. SovietFamine1932 has no row in that table.

## China, 1954–1966

Lin and Yang, Table 1, gives national grain output, procurement, the procurement share of output, net grain export, and per capita grain for every year 1954 through 1966. The table cites the State Statistical Bureau (1990) and the Ministry of Agriculture Planning Bureau (1984). Those two volumes were not opened, so this is one transcription, not a second official check. Provincial output is not in Table 1.

Output, in million tons: 1954 169.52, 1955 183.74, 1956 192.75, 1957 195.05, 1958 200.00, 1959 170.00, 1960 143.50, 1961 147.50, 1962 160.00, 1963 170.00, 1964 187.50, 1965 194.53, 1966 214.00.

The registry window is 1959–1961. Output in 1958 is the high point in this table. Output in 1960 is 143.50, and output in 1966 is the first later year above 200. The paper's comparison with 1958 is a comparison inside this table. It is not a causal estimate. Procurement is separate: 64.12 million tons in 1959, against output of 170.00. Net export is 4.2 million tons in 1959 and −4.5 in 1961. Per capita grain is not a production loss.

The 1961 FAOSTAT cereals row linked in the observation file is a different series. It is not this 147.50 million tons.

## Soviet harvest evaluations, 1920–1923

Wheatcroft's thesis, on the page labeled 478, places three early TsSU evaluations side by side for USSR less SCR. That is his smaller area, excluding Transcaucasia and Turkestan, and sometimes parts of the Far East. It is not the full Soviet Union, and it is not the Volga.

| Year | No correction | Half-point correction | July 1923 evaluation |
|---|---:|---:|---:|
| 1920 | 29.00 | 34.58 | 41.9 |
| 1921 | 23.46 | 27.67 | 35.2 |
| 1922 | 30.51 | 36.23 | 45.7 |
| 1923 | 29.24 | 35.89 | 48.5 |

Units are million tons, as printed. On the page labeled 474, source block 3a totals 34.11 in 1920 and 27.67 in 1921 for the same smaller area. The 1920 half-point figure on page 478 is 34.58, so 34.11 and 34.58 are both kept. The 1921 half-point figure and the block 3a total are the same 27.67. Block 3a also splits 1920–1922 across the Northern Consumer Region, the Southern Producer Region, the Central Producer Region, and EPR. SCR is blank in that block and is not entered as zero. In 1921 the Southern Producer Region is 12.51 and the Central Producer Region is 5.01, against 16.23 and 5.99 in 1920. That is a regional record in one source block, not a Volga loss and not the July 1923 total of 35.2. Years 1918 and 1919 were not extracted.

## Ireland, potato area

The CSO Potatoes section restates Bourke. Area planted, in hectares as the CSO prints them: 785,308 in 1844, 829,875 in 1845, 656,014 in 1846, 89,000 in 1847, and 258,000 in 1848. The CSO says the 1844 and 1845 figures were collected retrospectively. The 1859 area in the same state-level series is 374,000 ha, which is after the registry window. Annual area for 1849–1852, yield, and oats were not on the pages read. The CSO also restates Bourke's pre-famine balance sheet as 15 million tons, with 47 percent human food, 33 percent animal feed, 13 percent seed, 2 percent exports, and 5 percent losses or waste. That sheet has no calendar year. The 5 percent is a balance-sheet residual, not a blight loss. The unit is tons as printed. Potato is not entered as grain.

## Not re-opened

**Drought18761878.** The target is district and taluk crop-outturn returns. The GIPE file did not open on retrieval. The Tanjore and Coimbatore percentages remain in the famine inventory and the search log. They were not copied into this table. The Gazetteer figure of 83,800 square miles remains famine area.

**SovietFamine1932.** The approximate harvest statements near 56 and 50 million tonnes were not re-read. They are not in the observation table. Ukraine and Kazakhstan are not separated. This case is not the 1921 reconstruction.

## Outside this frame

GreatFamine, Laki1783, NoSummer, and DustBowl had no collection in the four-source frame named by the famine inventory. NoSummer stays a qualitative note in the evidence seed. DustBowl has no Great Plains harvest series here. The 1934 national wheat level and the decadal model stay in the evidence seed.

## What this pass does not do

It does not append these rows to FAOSTAT. It does not change the registry, the envelopes, or the figures. It does not turn a national grain series into a provincial loss, or a potato area into cereal tonnes.
