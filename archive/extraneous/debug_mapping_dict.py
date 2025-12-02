import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_mapping_keys():
    # Manually check what's in the file content previously read
    # The previous read showed the 'complete_mapping' dict.
    # Let's verify if USA is there.
    
    complete_mapping = {
        'AFG': 'AF', 'AGO': 'AO', 'ALB': 'AL', 'ARE': 'TC', 'ARG': 'AR',
        'ARM': 'AM', 'ATG': 'AC', 'AUS': 'AS', 'AUT': 'AU', 'AZE': 'AJ',
        'BDI': 'BY', 'BEL': 'BE', 'BEN': 'BN', 'BFA': 'UV', 'BGD': 'BG',
        'BGR': 'BU', 'BHR': 'BA', 'BHS': 'BF', 'BIH': 'BK', 'BLR': 'BO',
        'BLZ': 'BH', 'BOL': 'BL', 'BRA': 'BR', 'BRB': 'BB', 'BRN': 'BX',
        'BTN': 'BT', 'BWA': 'BC', 'CAF': 'CT', 'CAN': 'CA', 'CHE': 'SZ',
        'CHL': 'CI', 'CHN': 'CH', 'CMR': 'CM', 'COD': 'CF', 'COG': 'CG',
        'COL': 'CO', 'CRI': 'CS', 'CSK': 'LO', 'CUB': 'CU', 'CYP': 'CY',
        'CZE': 'EZ', 'DEU': 'GM', 'DJI': 'DJ', 'DMA': 'DR', 'DNK': 'DA',
        'DOM': 'DO', 'DZA': 'AG', 'ECU': 'EC', 'EGY': 'EG', 'ERI': 'ER',
        'ESP': 'SP', 'EST': 'EN', 'ETH': 'ET', 'FIN': 'FI', 'FJI': 'FJ',
        'FRA': 'FR', 'GAB': 'GB', 'GBR': 'UK', 'GEO': 'GG', 'GHA': 'GH',
        'GIN': 'PU', 'GMB': 'GA', 'GNQ': 'GV', 'GRC': 'GR', 'GRD': 'GJ',
        'GTM': 'GT', 'GUY': 'GY', 'HND': 'HO', 'HRV': 'HR', 'HTI': 'HA',
        'HUN': 'HU', 'IDN': 'ID', 'IND': 'IN', 'IRL': 'EI', 'IRN': 'IR',
        'IRQ': 'IZ', 'ISL': 'IC', 'ISR': 'IS', 'ITA': 'IT', 'JAM': 'JM',
        'JOR': 'JO', 'JPN': 'JA', 'KAZ': 'KZ', 'KEN': 'KE', 'KGZ': 'KG',
        'KHM': 'CB', 'KNA': 'SC', 'KOR': 'KS', 'KWT': 'KU', 'LAO': 'LA',
        'LBN': 'LE', 'LBR': 'LI', 'LBY': 'LY', 'LCA': 'ST', 'LKA': 'CE',
        'LSO': 'LT', 'LTU': 'LH', 'LUX': 'LU', 'LVA': 'LG', 'MAR': 'MO',
        'MDA': 'MD', 'MDG': 'MA', 'MEX': 'MX', 'MKD': 'MK', 'MLI': 'ML',
        'MLT': 'MT', 'MMR': 'BM', 'MNE': 'MW', 'MNG': 'MG', 'MOZ': 'MZ',
        'MRT': 'MR', 'MUS': 'MP', 'MWI': 'MI', 'MYS': 'MY', 'NAM': 'WA',
        'NCL': 'NC', 'NER': 'NG', 'NGA': 'NI', 'NIC': 'NU', 'NLD': 'NL',
        'NOR': 'NO', 'NPL': 'NP', 'NZL': 'NZ', 'OMN': 'MU', 'PAK': 'PK',
        'PAN': 'PM', 'PER': 'PE', 'PHL': 'RP', 'PNG': 'PP', 'POL': 'PL',
        'PRI': 'PQ', 'PRK': 'KN', 'PRT': 'PO', 'PRY': 'PA', 'PSE': 'WE',
        'QAT': 'QA', 'ROU': 'RO', 'RUS': 'RS', 'RWA': 'RW', 'SAU': 'SA',
        'SDN': 'SU', 'SEN': 'SG', 'SLB': 'BP', 'SLE': 'SL', 'SLV': 'ES',
        'SOM': 'SO', 'SRB': 'RI', 'SSD': 'OD', 'STP': 'TP', 'SUR': 'NS',
        'SVK': 'LO', 'SVN': 'SI', 'SWE': 'SW', 'SWZ': 'WZ', 'SYR': 'SY',
        'TCD': 'CD', 'TGO': 'TG', 'THA': 'TH', 'TJK': 'TI', 'TKM': 'TX',
        'TLS': 'TT', 'TTO': 'TD', 'TUN': 'TS', 'TUR': 'TU', 'TZA': 'TZ',
        'UGA': 'UG', 'UKR': 'UP', 'URY': 'UY', 'USA': 'US', 'UZB': 'UZ',
        'VCT': 'VC', 'VEN': 'VE', 'VNM': 'VM', 'VUT': 'NH', 'YEM': 'YM',
        'ZAF': 'SF', 'ZMB': 'ZA', 'ZWE': 'ZI',
    }
    
    print(f"USA mapping: {complete_mapping.get('USA')}")

if __name__ == "__main__":
    check_mapping_keys()

