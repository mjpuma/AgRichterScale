# Primary source passes (November 27, 2026)

## Pass 1: the 12 events with no opened source

Each reading is in `ancillary/primary_source_readings_v1.csv`. A row records the page that was opened, where in it the number sits, the quantity and unit exactly as printed, the place and years, the baseline, and the causes the source names. Pages that failed to open are listed as `blocked` or `not_found` with no quantity. Nothing was adopted as an event loss. No observed field was filled, and the registry was not edited.

## What was opened

| Event | Source opened | What it says | Class after this pass |
|---|---|---|---|
| Ethiopia (1983–85) | USDA ERS, *Sub-Saharan Africa Outlook and Situation Report* RS-85-10, July 1985, PDF p. 15 | 1984 food grain 20–30 percent below normal, over 5 million tons. The report names rainfall, pests, government policy, and civil war | grain quantity |
| Syria (2011–) | UN News for FAO, 15 May 2014; FAO Newsroom, 9 October 2018 | 2014 wheat 1.97 Mt, about 52 percent below the 2001–2011 average. 2018 wheat 1.2 Mt, about two-thirds of 2017. Drought and conflict are named together | grain quantity |
| Yemen (2015–) | FAO GIEWS country brief, 10 February 2017 (archive PDF) | 2016 cereals about 480,000 t, 11 percent below 2015 and 37 percent below the five-year average. Conflict and the July 2016 floods are named | grain quantity |
| Haiti (2010–16) | FAO Newsroom, 22 September 2010, reporting the FAO/WFP CFSAM | 2010 cereals forecast at 503,600 t, 9 percent below 2009. Earthquake damage and late rains are named. The mission report itself was not opened | grain quantity |
| Laos (2015–16) | FAO GIEWS country brief, 24 May 2016 | 2015 rice 3.2 Mt, 4 percent below the 2014 record. Maize 1.1 Mt, 5 percent below 2014. El Niño rains are named | grain quantity |
| EastTimor (2015–16) | FAO GIEWS country brief, 14 December 2016 | 2016 rice 60,000 t, 6 percent below 2015 and 30 percent below average. Maize 60,000 t, 8 and 21 percent below. El Niño dry weather is named | grain quantity |
| SierraLeone (1991–2002) | FAO GIEWS special report, 4 February 2000 | 1999 paddy 248,220 t, about 45 percent of 1990 (543,700 t). The report calls the estimates provisional | grain quantity |
| Liberia (1989–2003) | FAO GIEWS missions, 16 January 1996 and 22 December 2000 | 1995 paddy at most 23 percent of pre-war, which the mission calls highly tentative. 2000 paddy 144,000 t against 259,000 t in 1988. No official crop data exists after 1989 | grain quantity |
| SahelDrought2010 | FAO Sahel Information Note 1, 6 February 2012, p. 1 | CILSS Sahel cereals 16.6 Mt in 2011/12, 25 percent below the 2010/11 record and 2 percent below five years. Chad, Mauritania, and Senegal fell 50, 52, and 31 percent from 2010/11 | production context (the aggregate is not the 10 workbook countries) |
| Bangladesh (1974) | Dyson, IUSSP 2021 conference paper, PDF p. 8 | The end-1974 aman crop was 700,000 tons below 1973, tied to the floods. The sentence names no data source | no usable quantity |
| Laki1783 | Wieners and Hálfdanarson, NHESS 24 (2024) | No crop quantity. Haze damaged Icelandic grass, and records show no harvest failure in west and central Europe in 1783 | no usable quantity |
| Solomon (2015–16) | none found | Searches turned up only qualitative drought reports, and GIEWS has no Solomon Islands brief | no source |

## How to read these numbers

- Each figure is national production, or for Sahel a regional aggregate, compared against the baseline the source states. None is an event loss or a loss fraction. Every page that names a cause names more than one.
- The baselines differ: the previous year, a five-year average, a pre-war year, or an undefined "normal". They cannot be compared with each other or with the SPAM 2020 exposure on the figures.
- The FAO GIEWS and mission estimates for Liberia and Sierra Leone were made without national surveys, and the reports say so.
- These readings were not appended to the FAOSTAT panel or the archival table. A FAOSTAT row for the same country and year is a separate series.
- The event-classification rule is unchanged. Moving into the `grain_quantity_opened` class means an opened page now gives a grain quantity for the event's place and years. It does not mean the event caused that change.

## Blocked or not found

ReliefWeb (UNDRO Ethiopia 1984; the GIEWS Yemen mirror), UNU-WIDER WP26, and FAO Open Knowledge (the Timor-Leste El Niño project, which a search snippet says gives maize −13 and rice −71 percent) all returned HTTP 403. The CIA reading-room node for the Bangladesh foodgrain outlook served no document. None of these is used.

## Pass 2: the other 15 events

These events already had opened sources from earlier passes: the source check, the archival reconstruction, the evidence seed, and the documentary log. This pass looked for agency or primary pages that had not yet been opened.

| Event | Source opened | What it says | Class after this pass |
|---|---|---|---|
| NorthKorea1990s | FAO/WFP CFSAM special reports, 6 December 1996 and 25 November 1997 | 1996 grain about 4.3 Mt, after 300,000 t of flood losses. 1997 maize 1.138 Mt, 51 percent below 1996 and the lowest on record. 1997 rice 1.52 Mt milled. The yields are mission estimates | grain quantity (was a contested claim) |
| MillenniumDrought | ABS 7121.0, *Agricultural Commodities, Australia, 2002-03*, released 28 June 2004 | 2002-03 wheat fell 58 percent to 10.1 Mt. NSW fell 69 percent to 2.5 Mt. The baseline is not named | grain quantity for one season (was a yield gap) |
| UkraineWar2022 | FAO GIEWS country brief, 16 December 2022 | 2022 cereals 51 Mt, almost 30 percent below the five-year average and 40 percent below the exceptional 2021 harvest. The brief names the war's effect on fields, labour, and costs | grain quantity (was production context) |
| ENSO2015_2016 | FAO Southern Africa Situation Report, September 2016 | Regional cereal shortfall of nearly 9.3 Mt in 2015/16. Southern Africa is not on the ENSO workbook sheet | production context (unchanged) |

Blocked or not tried:
- Goodkind and West (2001): Wiley returned HTTP 403, and JSTOR served a browser challenge.
- Tauger on the 1932 harvest: the Pitt page went over the fetch size limit, and the WVU host refused the connection.
- Jordan (1996), Stommel and Stommel (1983), Bourke (1960), the Indian Famine Commission report (1880), and Titow are print books or paywalled papers. No open copy was tried.
- GreatFamine, NoSummer, PotatoFamine, Drought18761878, SovietFamine1921, SovietFamine1932, ChineseFamine1960, DustBowl, Vanuatu, SudanConflict2023, and SouthSudanConflict gained no new reading in this pass. Their classes are unchanged.

The MillenniumDrought figure covers one season of a 2000–2009 registry window. It sits alongside van Dijk (2013), who reports that production over the whole drought rose as more area was planted. Neither number is an event loss.
