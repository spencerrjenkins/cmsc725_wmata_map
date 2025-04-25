'''url_prefix = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
url_suffix = "" + url_suffix
url = url_prefix + "Health_WebMercator/MapServer/4" + url_suffix
hospital_df = get_data(url, 'hospital')
url = url_prefix + "Business_Goods_and_Service_WebMercator/MapServer/4" + url_suffix
grocery_df = get_data(url, 'grocery')
url = url_prefix + "Cultural_and_Society_WebMercator/MapServer/5" + url_suffix
religion_df = get_data(url, 'religion')
url = url_prefix + "Cultural_and_Society_WebMercator/MapServer/4" + url_suffix
library_df = get_data(url, 'library')
url = url_prefix + "Location_WebMercator/FeatureServer/3" + url_suffix
poi_df = get_data(url, 'poi')
url = url_prefix + "Education_WebMercator/MapServer/5" + url_suffix
pub_school_df = get_data(url, 'pub_school')
url = url_prefix + "Education_WebMercator/MapServer/3" + url_suffix
ind_school_df = get_data(url, 'ind_school')
url = url_prefix + "Recreation_WebMercator/MapServer/3" + url_suffix
rec_df = get_data(url, 'rec')
url = url_prefix + "Cultural_and_Society_WebMercator/MapServer/54" + url_suffix
museum_df = get_data(url, 'museum')
url = url_prefix + "Cultural_and_Society_WebMercator/MapServer/6" + url_suffix
unknown_df = get_data(url, 'unknown')
url = url_prefix + "Health_WebMercator/MapServer/7" + url_suffix
primary_care_df = get_data(url, 'primary_care')
url = url_prefix + "Public_Service_WebMercator/MapServer/55" + url_suffix
wic_df = get_data(url, 'wic')
url = url_prefix + "Business_Goods_and_Service_WebMercator/MapServer/9" + url_suffix
pharm_df = get_data(url, 'pharm')'''
'''url_prefix = "https://geodata.md.gov/imap/rest/services/"
url_suffix = "/query?where=1%3D1&outFields=*&outSR=4326&f=json"
url = url_prefix + "Structure/MD_StateFacilities/FeatureServer/0" + url_suffix
state_facilities_df = get_data(url, 'state_facilities', 'md')
url = url_prefix + "Military/MD_MilitaryInstallations/FeatureServer/0" + url_suffix
fed_mil_df = get_data(url, 'fed_mil', 'md')
url = url_prefix + "Military/MD_MilitaryInstallations/FeatureServer/1" + url_suffix
state_mil_df = get_data(url, 'state_mil', 'md')
url = url_prefix + "Structure/MD_CommunitySupport/FeatureServer/0" + url_suffix
gov_support_df = get_data(url, 'gov_support', 'md')
url = url_prefix + "BusinessEconomy/MD_IncentiveZones/FeatureServer/11" + url_suffix
incentive_zones_df = get_data(url, 'incentive_zones', 'md')
url = url_prefix + "BusinessEconomy/MD_IncentiveZones/FeatureServer/1" + url_suffix
main_street_df = get_data(url, 'main_street', 'md')
url = url_prefix + "Education/MD_Libraries/FeatureServer/0" + url_suffix
libraries_df = get_data(url, 'libraries', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/2" + url_suffix
fy_priv_df = get_data(url, 'fy_priv', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/5" + url_suffix
k12_public_df = get_data(url, 'k12_public', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/6" + url_suffix
k12_charter_df = get_data(url, 'k12_charter', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/4" + url_suffix
higher_ed_df = get_data(url, 'higher_ed', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/1" + url_suffix
ty_public_df = get_data(url, 'ty_public', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/0" + url_suffix
fy_public_df = get_data(url, 'fy_public', 'md')
url = url_prefix + "Education/MD_EducationFacilities/FeatureServer/3" + url_suffix
ty_private_df = get_data(url, 'ty_private', 'md')
url = url_prefix + "Health/MD_LongTermCareAssistedLiving/FeatureServer/1" + url_suffix
assisted_df = get_data(url, 'assisted', 'md')
url = url_prefix + "Health/MD_Hospitals/FeatureServer/0" + url_suffix
hospital_df = get_data(url, 'hospital', 'md')
url = url_prefix + "Historic/MD_NationalRegisterHistoricPlaces/FeatureServer/0" + url_suffix
nrhp_df = get_data(url, 'nrhp', 'md')
url = url_prefix + "BusinessEconomy/MD_IncentiveZones/FeatureServer/12" + url_suffix
arts_df = get_data(url, 'arts', 'md')'''
'''combined_df = pd.concat([hospital_df, grocery_df, religion_df, library_df, poi_df, pub_school_df, ind_school_df, rec_df, museum_df, unknown_df, primary_care_df, wic_df, pharm_df])
combined_df.reset_index(inplace=True, drop=True)
combined_df.to_file("data/dc/non-population-points/combined_df.geojson", driver="GeoJSON")'''
import re
combined_df = pd.concat([state_facilities_df, fed_mil_df, state_mil_df, gov_support_df, incentive_zones_df, main_street_df, libraries_df, fy_priv_df, k12_public_df, k12_charter_df, higher_ed_df, ty_public_df, fy_public_df, ty_private_df, assisted_df, hospital_df, nrhp_df])
def converter(a):
    if type(a) == dict:
        if 'county' in a:
            return a['county'].strip().lower()
        elif "County" in a:
            return a['County'].strip().lower()
        else:
            return 'baltimore city'
    else:
        try:
            return converter(dict(a))
        except:
            try:
                return converter(json.loads(a))
            except:
                matches = re.finditer(r'county[\'"] ?:', a.lower())
                for match in matches:
                    search_area = a.lower()[match.end():match.end()+50].strip()
                    m = re.findall(r'[\'"]([a-z \']+)[\'"]', search_area)
                    if m and m[0]:
                        return m[0]
                return 'UNK'
combined_df["county"] = combined_df["attributes"].apply(converter)
combined_df = combined_df[combined_df["county"].apply(lambda a: 'george' in a or 'montgom' in a)]
combined_df.reset_index(inplace=True, drop=True)
combined_df.to_file("data/md/non-population-points/combined_df.geojson", driver="GeoJSON")
combined_df = gpd.GeoDataFrame.from_file("data/dc/non-population-points/combined_df.geojson")
combined_df