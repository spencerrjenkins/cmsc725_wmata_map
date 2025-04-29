import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon

def get_data(url, name, state):
    
    try:
        return gpd.GeoDataFrame.from_file(f"data/{state}/non-population-points/{name}.geojson")
    except:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            df = gpd.GeoDataFrame(data["features"])
            df = df.set_geometry(df["geometry"].apply(lambda a: Point(a.values()) if len(a.keys()) == 2 else Polygon(a["rings"][0]).centroid), crs="EPSG:4326")
            df.to_crs("EPSG:4326",inplace=True)
            df.to_file(f"data/{state}/non-population-points/{name}.geojson", driver="GeoJSON")
            return df
        else:
            print("Error fetching crime data:", response.status_code)
            return gpd.GeoDataFrame()
