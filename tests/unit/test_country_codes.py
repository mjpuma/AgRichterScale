"""Unit tests for country-code mappings.

Guards against the FIPS-guessing bug class: FIPS 10-4 codes differ from ISO2
in surprising ways (Nigeria=NI but Niger=NG; Germany=GM; Ireland=EI;
Switzerland=SZ; Australia=AS), so any heuristic that derives FIPS from ISO3
letters silently attributes one country's agriculture to another.
"""

from pathlib import Path

import pandas as pd
import pytest

from agrichter.data.spatial_mapper import ISO3_TO_FIPS_STATIC

ANCILLARY = Path(__file__).parent.parent.parent / 'ancillary'


class TestStaticIso3ToFips:
    """Spot-check the authoritative mapping against known FIPS 10-4 codes."""

    KNOWN_CORRECT = {
        # The famous traps (first-two-letters heuristic fails on all of these)
        'IRL': 'EI',   # Ireland (NOT 'IR' = Iran)
        'CHE': 'SZ',   # Switzerland (NOT 'CH' = China)
        'GBR': 'UK',   # United Kingdom (NOT 'GB' = Gabon)
        'DEU': 'GM',   # Germany (NOT 'DE'; 'GE' is not Germany either)
        'AUS': 'AS',   # Australia (NOT 'AU' = Austria)
        'AUT': 'AU',   # Austria
        'NGA': 'NI',   # Nigeria (FIPS swaps Nigeria/Niger relative to ISO2)
        'NER': 'NG',   # Niger
        'SWE': 'SW',   # Sweden
        'ESP': 'SP',   # Spain
        'PRT': 'PO',   # Portugal
        'IRQ': 'IZ',   # Iraq
        'IRN': 'IR',   # Iran
        'JPN': 'JA',   # Japan
        'KOR': 'KS',   # South Korea
        'PRK': 'KN',   # North Korea
        'CHN': 'CH',   # China
        'CAF': 'CT',   # Central African Republic
        'TCD': 'CD',   # Chad
        'USA': 'US',
        'IND': 'IN',
        'BRA': 'BR',
        'FRA': 'FR',
        'EGY': 'EG',
        'ARG': 'AR',
        'RUS': 'RS',   # Russia (NOT 'RU')
        'UKR': 'UP',   # Ukraine (NOT 'UA')
        'ZAF': 'SF',   # South Africa (NOT 'ZA' = Zambia in FIPS)
        'ZMB': 'ZA',   # Zambia
        'COD': 'CG',   # DR Congo (FIPS 'CG' = Congo Kinshasa; ISO2 CD is Chad in FIPS!)
        'COG': 'CF',   # Congo-Brazzaville (FIPS 'CF' = Congo Brazzaville)
        'DMA': 'DO',   # Dominica (NOT 'DR' = Dominican Republic)
        'DOM': 'DR',   # Dominican Republic
        'GIN': 'GV',   # Guinea (NOT 'PU' = Guinea-Bissau)
        'GNB': 'PU',   # Guinea-Bissau
        'GNQ': 'EK',   # Equatorial Guinea (NOT 'GV')
        'TLS': 'TT',   # Timor-Leste
        'TMP': 'TT',   # Timor-Leste (old ISO3 used by CountryCode_Convert.xls)
    }

    @pytest.mark.parametrize('iso3,fips', sorted(KNOWN_CORRECT.items()))
    def test_known_mapping(self, iso3, fips):
        assert ISO3_TO_FIPS_STATIC.get(iso3) == fips

    def test_no_iso3_maps_to_itself_truncated_wrongly(self):
        """The old bug: FIPS guessed as iso3[:2]. Verify the known traps differ.
        (Sweden is excluded: SWE -> SW genuinely matches its first two letters.)"""
        traps = ['IRL', 'CHE', 'GBR', 'DEU', 'AUS', 'NGA', 'ESP', 'PRT']
        for iso3 in traps:
            assert ISO3_TO_FIPS_STATIC[iso3] != iso3[:2], (
                f"{iso3}: static map equals the first-two-letter guess; "
                f"this test list needs updating if that is genuinely correct"
            )

    def test_values_are_two_letter_codes(self):
        for iso3, fips in ISO3_TO_FIPS_STATIC.items():
            assert len(iso3) == 3 and iso3.isupper()
            assert len(fips) == 2 and fips.isupper()


class TestStaticMappingAgainstSpam:
    """Verify mapped FIPS codes actually exist in the SPAM 2020 grid."""

    @pytest.fixture(scope='class')
    def spam_fips(self):
        spam_csv = (Path(__file__).parent.parent.parent
                    / 'spam2020V2r0_global_harvested_area'
                    / 'spam2020V2r0_global_harvested_area'
                    / 'spam2020V2r0_global_H_TA.csv')
        if not spam_csv.exists():
            pytest.skip("SPAM CSV not available")
        df = pd.read_csv(spam_csv, usecols=['FIPS0', 'ADM0_NAME'])
        pairs = df.drop_duplicates()
        return dict(zip(pairs['FIPS0'], pairs['ADM0_NAME']))

    NAME_CHECKS = {
        'IRL': 'Ireland',
        'CHE': 'Switzerland',
        'DEU': 'Germany',
        'AUS': 'Australia',
        'NGA': 'Nigeria',
        'NER': 'Niger',
        'CHN': 'China',
        'USA': 'United States',
        'CAF': 'Central African',
        'TCD': 'Chad',
        'EGY': 'Egypt',
        'ZAF': 'South Africa',
        'COD': 'Democratic Republic of the Congo',
        'DMA': 'Dominica',
        'DOM': 'Dominican Republic',
        'GIN': 'Guinea',
        'GNB': 'Guinea-Bissau',
        'GNQ': 'Equatorial Guinea',
        'TMP': 'Timor',
    }

    @pytest.mark.parametrize('iso3,name_fragment', sorted(NAME_CHECKS.items()))
    def test_fips_resolves_to_expected_country_name(self, spam_fips, iso3, name_fragment):
        fips = ISO3_TO_FIPS_STATIC[iso3]
        assert fips in spam_fips, f"FIPS {fips} ({iso3}) absent from SPAM grid"
        assert name_fragment.lower() in spam_fips[fips].lower(), (
            f"{iso3} -> FIPS {fips} resolves to '{spam_fips[fips]}', "
            f"expected something like '{name_fragment}'"
        )

    def test_most_static_fips_exist_in_spam(self, spam_fips):
        """The vast majority of mapped codes must exist in the SPAM grid
        (a few tiny islands/city-states legitimately have no crop cells)."""
        missing = [f"{i}->{f}" for i, f in ISO3_TO_FIPS_STATIC.items()
                   if f not in spam_fips]
        assert len(missing) <= 10, f"Too many mapped FIPS absent from SPAM: {missing}"


class TestGdamCodes:
    """Verify numeric GDAM codes in the conversion table for key countries."""

    @pytest.fixture(scope='class')
    def convert_table(self):
        xls = ANCILLARY / 'CountryCode_Convert.xls'
        if not xls.exists():
            pytest.skip("CountryCode_Convert.xls not available")
        return pd.read_excel(xls)

    def test_china_gdam_is_48_not_45(self, convert_table):
        """Regression: a test once used 45 for China; 45 is Central African Republic."""
        row = convert_table[convert_table['Country'] == 'China']
        assert len(row) == 1
        assert int(row.iloc[0]['GDAM ']) == 48
        car = convert_table[convert_table['GDAM '] == 45]
        assert 'Central African' in car.iloc[0]['Country']

    def test_usa_gdam_is_240_not_231(self, convert_table):
        """Regression: a test once used 231 for USA; 231 is Tunisia.
        The event sheets (e.g. DustBowl) correctly use 240."""
        usa = convert_table[convert_table['GDAM '] == 240]
        assert len(usa) >= 1
        assert 'United States' in usa.iloc[0]['Country']
        tunisia = convert_table[convert_table['GDAM '] == 231]
        assert tunisia.iloc[0]['Country'] == 'Tunisia'

    def test_event_sheet_codes_resolve_to_expected_countries(self, convert_table):
        """Every GDAM code used in the historical event sheets must resolve
        to the country named in that sheet's own Country column."""
        xls = ANCILLARY / 'DisruptionCountry.xls'
        if not xls.exists():
            pytest.skip("DisruptionCountry.xls not available")
        gdam_to_name = dict(zip(convert_table['GDAM '], convert_table['Country']))
        # Verified naming variants: sheet name -> fragment of the table name
        aliases = {
            'Burina Faso': 'Burkina Faso',   # typo in sheet, code 37 is correct
            'North Korea': "Democratic People's Republic of Korea",
            'East Timor': 'Timor-Leste',
            'Laos': 'Lao People',
        }
        sheets = pd.read_excel(xls, sheet_name=None, engine='xlrd')
        mismatches = []
        for event, df in sheets.items():
            code_col = df.columns[1]  # second column holds the GDAM code
            for _, row in df.iterrows():
                sheet_name = str(row['Country']).strip()
                code = row[code_col]
                if pd.isna(code):
                    continue
                table_name = str(gdam_to_name.get(int(code), 'MISSING'))
                expected = aliases.get(sheet_name, sheet_name)
                # Compare on the first significant word (handles
                # 'Russian Federation' vs 'Russia' style variations)
                if expected.split()[0].lower()[:5] not in table_name.lower():
                    mismatches.append(f"{event}: '{sheet_name}' code {code} -> '{table_name}'")
        assert not mismatches, "Event sheet GDAM codes mismatch:\n" + "\n".join(mismatches)


class TestEventSheetParsing:
    """Regression tests for the event Excel sheet parser."""

    def test_country_sheet_keeps_first_row(self):
        """Regression: a former `iloc[1:]` skip silently dropped the FIRST
        country of every multi-country event (e.g. Australia from ENSO
        2015-16, Burkina Faso from the 2010 Sahel drought)."""
        from agrichter.core.config import Config
        from agrichter.data.events import EventsProcessor

        processor = EventsProcessor(Config(crop_type='allgrain', root_dir='.'))
        df = pd.DataFrame({
            'Country': ['Australia', 'Brazil', 'Brunei'],
            'ENSO 2015-2016': [14, 32, 35],
            'state flag': [1, 1, 0],
        })
        result = processor._process_country_sheet(df)
        assert result['countries'] == ['Australia', 'Brazil', 'Brunei']
        assert result['country_codes'] == [14.0, 32.0, 35.0]
        assert result['state_flags'] == [1.0, 1.0, 0.0]

    def test_enso_sheet_includes_australia(self):
        """End-to-end check on the real Excel file."""
        xls = ANCILLARY / 'DisruptionCountry.xls'
        if not xls.exists():
            pytest.skip("DisruptionCountry.xls not available")
        from agrichter.core.config import Config
        from agrichter.data.events import EventsProcessor

        processor = EventsProcessor(Config(crop_type='allgrain', root_dir='.'))
        sheets = pd.read_excel(xls, sheet_name=None, engine='xlrd')
        result = processor._process_country_sheet(sheets['ENSO2015_2016'])
        assert 'Australia' in result['countries']
        assert 14.0 in result['country_codes']
