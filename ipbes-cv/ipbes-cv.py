"""IPBES global coastal vulnerability calculation."""
import gzip
import pandas
import sys
import traceback
import shutil
import time
import os
import math
import logging
import re
import multiprocessing
import zipfile

import google.cloud.client
import google.cloud.storage
import reproduce
import reproduce.utils
import taskgraph
import numpy
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import rtree
import shapely
import shapely.wkb
import shapely.ops
import shapely.speedups
import pygeoprocessing

# set a 1GB limit for the cache
gdal.SetCacheMax(2**30)

logging.basicConfig(
    format='%(asctime)s %(name)-10s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ',
    stream=sys.stdout)
LOGGER = logging.getLogger('ipbes-cv')
LOGGER.setLevel(logging.DEBUG)

POSSIBLE_DROPBOX_LOCATIONS = [
    r'D:\Dropbox\ipbes_old',
    r'C:\Users\Rich\Dropbox\ipbes_old',
    r'C:\Users\rpsharp\Dropbox\ipbes_old',
    r'E:\Dropbox\ipbes_old',
    r'./Dropbox/ipbes_old']

LOGGER.info("checking dropbox locations")
for path in POSSIBLE_DROPBOX_LOCATIONS:
    LOGGER.debug(path)
    if os.path.exists(path):
        BASE_DROPBOX_DIR = path
        break
LOGGER.info("found %s", BASE_DROPBOX_DIR)

try:
    IAM_TOKEN_PATH = os.path.normpath(sys.argv[1])
except IndexError:
    raise RuntimeError("Expected command line argument of path to bucket key")

_N_CPUS = -1 #max(1, multiprocessing.cpu_count())

WORKING_DIR = "ipbes_cv_workspace_fixed_global_geom"
ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
_TARGET_NODATA = -1
_GLOBAL_WWIII_GS_PATH = 'gs://ipbes-natcap-ecoshard-data-for-publication/wave_watch_iii_md5_c8bb1ce4739e0a27ee608303c217ab5b.gpkg.gz'
_GLOBAL_DEM_GS_PATH = 'gs://ipbes-natcap-ecoshard-data-for-publication/global_dem_md5_22c5c09ac4c4c722c844ab331b34996c.tif'
_TM_WORLD_BORDERS_GS_PATH = 'gs://ipbes-natcap-ecoshard-data-for-publication/TM_WORLD_BORDERS_SIMPL-0.3_md5_c0d1b65f6986609031e4d26c6c257f07.gpkg'
_GLOBAL_POLYGON_GS_PATH = 'gs://ipbes-natcap-ecoshard-data-for-publication/ipbes-cv_global_polygon_simplified_geometries_md5_653118dde775057e24de52542b01eaee.gpkg'


# layer name, (layer path, layer rank, protection distance)
_GLOBAL_HABITAT_LAYER_PATHS = {
    'mangrove': (('gs://ipbes-natcap-ecoshard-data-for-publication/ipbes-cv_mangrove_valid_md5_8f54e3ed5eb3b4d183ce2cd6ebbe480d.gpkg'), 1, 1000.0),
    'saltmarsh': (('gs://ipbes-natcap-ecoshard-data-for-publication/ipbes-cv_saltmarsh_valid_md5_56364edc15ab96d79b9fa08b12ec56ab.gpkg'), 2, 1000.0),
    'coralreef': (('gs://ipbes-natcap-ecoshard-data-for-publication/ipbes-cv_coralreef_valid_md5_ddc0b3f7923f1ed53b3e174659158cfc.gpkg'), 1, 2000.0),
    'seagrass': (('gs://ipbes-natcap-ecoshard-data-for-publication/ipbes-cv_seagrass_valid_md5_e206dde7cc9b95ba9846efa12b63d333.gpkg'), 4, 500.0),
}

# tuple form is (path, divide by area?, area to search, extra pixels to add)
_AGGREGATION_LAYER_MAP = {
    'pdn_gpw': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/gpw-v4-population-count-2015/gpw-v4-population-count_2015.tif"), True, None, 1e3, 0),
    'pdn_ssp1': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/Spatial_population_scenarios_GeoTIFF/SSP1_GeoTIFF/total/GeoTIFF/ssp1_2050.tif"), True, None, 1e3, 0),
    'pdn_ssp3': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/Spatial_population_scenarios_GeoTIFF/SSP3_GeoTIFF/total/GeoTIFF/ssp3_2050.tif"), True, None, 1e3, 0),
    'pdn_ssp5': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/Spatial_population_scenarios_GeoTIFF/SSP5_GeoTIFF/total/GeoTIFF/ssp5_2050.tif"), True, None, 1e3, 0),
    'pdn_2010': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/Spatial_population_scenarios_GeoTIFF/SSP1_GeoTIFF/total/GeoTIFF/ssp1_2010.tif"), True, None, 1e3, 0),
    '14bt_pop': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/gpw_v4_e_a000_014bt_2010_cntm_30_sec.tif"), True, None, 1e3, 0),
    '65plus_pop': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/gpw_v4_e_a065plusbt_2010_cntm_30_sec.tif"), True, None, 1e3, 0),
    'urbp_2015': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/GLOBIO4_landuse_10sec_tifs_20171207_Idiv/Current2015/Globio4_landuse_10sec_2015_cropint.tif"), False, [1, 190], 5e3, 0),
    'urbp_ssp1': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/GLOBIO4_landuse_10sec_tifs_20171207_Idiv/SSP1_RCP26/Globio4_landuse_10sec_2050_cropint.tif"), False, [1, 190], 5e3, 0),
    'urbp_ssp3': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/GLOBIO4_landuse_10sec_tifs_20171207_Idiv/SSP3_RCP70/Globio4_landuse_10sec_2050_cropint.tif"), False, [1, 190], 5e3, 0),
    'urbp_ssp5': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/GLOBIO4_landuse_10sec_tifs_20171207_Idiv/SSP5_RCP85/Globio4_landuse_10sec_2050_cropint.tif"), False, [1, 190], 5e3, 0),
    'SLRrate_cur': (("gs://ipbes-natcap-ecoshard-data-for-publication/MSL_Map_MERGED_Global_AVISO_NoGIA_Adjust_md5_3072845759841d0b2523d00fe9518fee.tif"), False, None, 5e3, 1),
    'slr_rcp26': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/slr_rcp26_md5_7c73cf8a1bf8851878deaeee0152dcb6.tif"), False, None, 5e3, 2),
    'slr_rcp60': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/slr_rcp60_md5_99ccaf1319d665b107a9227f2bbbd8b6.tif"), False, None, 5e3, 2),
    'slr_rcp85': ((r"gs://ipbes-natcap-ecoshard-data-for-publication/slr_rcp85_md5_3db20b7e891a71e23602826179a57e4a.tif"), False, None, 5e3, 2),
}

# The global bounding box to do the entire analysis
# This range was roughly picked to avoid the poles
# [min_lat, min_lng, max_lat, max_lng]
_GLOBAL_BOUNDING_BOX_WGS84 = [-180, -60, 180, 60]

# This is the lat/lng grid size to slice the runs into, annoying since it's
# lat/lng, but if you have a better idea lets hear it.
# The 3.0 degrees comes from the fact that UTM zones are 6 degrees wide so
# half of that plus some buffer should be good enough
_WGS84_GRID_SIZE = 3.0

# Wave Watch III data does not cover the planet.  Make sure we don't deal
# with a point that's not in range of said point.  I'm picking 1 degree since
# that's double the diagonal distance between two WWIII points
_MAX_WWIII_DISTANCE = 5.0

# Choosing just 16 fetch rays to make things run faster
_N_FETCH_RAYS = 16

_GLOBAL_GRID_VECTOR_FILE_PATTERN = 'global_grid.gpkg'
_LANDMASS_BOUNDING_RTREE_FILE_PATTERN = 'global_feature_index.dat'
_GLOBAL_WWIII_RTREE_FILE_PATTERN = 'wwiii_rtree.dat'
_GRID_POINT_FILE_PATTERN = 'grid_points_%d.gpkg'
_WIND_EXPOSURE_POINT_FILE_PATTERN = 'rei_points_%d.gpkg'
_WAVE_EXPOSURE_POINT_FILE_PATTERN = 'wave_points_%d.gpkg'
_HABITAT_PROTECTION_POINT_FILE_PATTERN = 'habitat_protection_points_%d.gpkg'
_RELIEF_POINT_FILE_PATTERN = 'relief_%d.gpkg'
_SURGE_POINT_FILE_PATTERN = 'surge_%d.gpkg'
_SEA_LEVEL_POINT_FILE_PATTERN = 'sea_level_%d.gpkg'
_GLOBAL_REI_POINT_FILE_PATTERN = 'global_rei_points.gpkg'
_GLOBAL_WAVE_POINT_FILE_PATTERN = 'global_wave_points.gpkg'
_GLOBAL_RELIEF_POINT_FILE_PATTERN = 'global_relief_points.gpkg'
_GLOBAL_HABITAT_PROTECTION_FILE_PATTERN = (
    'global_habitat_protection_points.gpkg')
_GLOBAL_SURGE_POINT_FILE_PATTERN = 'global_surge_points.gpkg'
_GLOBAL_SEA_LEVEL_POINT_FILE_PATTERN = 'global_sea_level_points.gpkg'
_GLOBAL_FETCH_RAY_FILE_PATTERN = 'global_fetch_rays.gpkg'
_GLOBAL_RISK_RESULT_POINT_VECTOR_FILE_PATTERN = 'CV_outputs.gpkg'
_AGGREGATE_POINT_VECTOR_FILE_PATTERN = (
    'global_cv_risk_and_aggregate_analysis.gpkg')
_WORK_COMPLETE_TOKEN_PATH = os.path.join(
    WORKING_DIR, 'work_tokens')
_WIND_EXPOSURE_WORKSPACES = os.path.join(
    WORKING_DIR, 'wind_exposure_workspaces')
_WAVE_EXPOSURE_WORKSPACES = os.path.join(
    WORKING_DIR, 'wave_exposure_workspaces')
_HABITAT_PROTECTION_WORKSPACES = os.path.join(
    WORKING_DIR, 'habitat_protection_workspaces')
_RELIEF_WORKSPACES = os.path.join(
    WORKING_DIR, 'relief_workspaces')
_GRID_WORKSPACES = os.path.join(
    WORKING_DIR, 'grid_workspaces')
_SURGE_WORKSPACES = os.path.join(
    WORKING_DIR, 'surge_workspaces')
_SEA_LEVEL_WORKSPACES = os.path.join(
    WORKING_DIR, 'sea_level_workspaces')
_POPULATION_MASK_WORKSPACE = os.path.join(WORKING_DIR, 'population_masks')
_SMALLEST_FEATURE_SIZE = 2000
_MAX_FETCH_DISTANCE = 60000


def main():
    """Entry point."""
    logging.basicConfig(
        format='%(asctime)s %(name)-10s %(levelname)-8s %(message)s',
        level=logging.WARN, datefmt='%m/%d/%Y %H:%M:%S ')
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    task_graph = taskgraph.TaskGraph(
        _WORK_COMPLETE_TOKEN_PATH, _N_CPUS, reporting_interval=5.0)

    tm_world_borders_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(_TM_WORLD_BORDERS_GS_PATH))
    tm_world_borders_basedata_fetch_task = task_graph.add_task(
        func=reproduce.utils.google_bucket_fetch_and_validate,
        args=(
            _TM_WORLD_BORDERS_GS_PATH, IAM_TOKEN_PATH,
            tm_world_borders_path),
        target_path_list=[tm_world_borders_path],
        task_name=f'fetch {os.path.basename(tm_world_borders_path)}')

    global_polygon_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(_GLOBAL_POLYGON_GS_PATH))
    global_polygon_fetch_task = task_graph.add_task(
        func=reproduce.utils.google_bucket_fetch_and_validate,
        args=(
            _GLOBAL_POLYGON_GS_PATH, IAM_TOKEN_PATH,
            global_polygon_path),
        target_path_list=[global_polygon_path],
        task_name=f'fetch {os.path.basename(global_polygon_path)}')

    global_dem_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(_GLOBAL_DEM_GS_PATH))
    global_dem_fetch_task = task_graph.add_task(
        func=reproduce.utils.google_bucket_fetch_and_validate,
        args=(
            _GLOBAL_DEM_GS_PATH, IAM_TOKEN_PATH,
            global_dem_path),
        target_path_list=[global_dem_path],
        task_name=f'fetch {os.path.basename(global_dem_path)}')

    global_wwiii_path = os.path.join(
        ECOSHARD_DIR, os.path.splitext(os.path.basename(
            _GLOBAL_WWIII_GS_PATH))[0])
    wwiii_fetch_task = task_graph.add_task(
        func=fetch_validate_and_unzip,
        args=(
            _GLOBAL_WWIII_GS_PATH, IAM_TOKEN_PATH, ECOSHARD_DIR,
            global_wwiii_path),
        target_path_list=[global_wwiii_path],
        task_name=f'fetch {os.path.basename(global_wwiii_path)}')

    wwiii_rtree_path = os.path.join(
        WORKING_DIR, _GLOBAL_WWIII_RTREE_FILE_PATTERN)
    build_wwiii_task = task_graph.add_task(
        func=build_wwiii_rtree,
        args=(global_wwiii_path, wwiii_rtree_path),
        dependent_task_list=[wwiii_fetch_task],
        task_name='build_wwiii_rtree')

    fetch_habitat_task_list = []
    habitat_vector_lookup = {}
    for habitat_id, (
            habitat_gs_url, habitat_rank, habitat_dist) in (
                _GLOBAL_HABITAT_LAYER_PATHS.items()):
        habitat_path = os.path.join(
            ECOSHARD_DIR, os.path.basename(habitat_gs_url))

        habitat_vector_lookup[habitat_id] = (
            habitat_path, habitat_rank, habitat_dist)

        # download habitat path
        fetch_habitat_blob_task = task_graph.add_task(
            func=reproduce.utils.google_bucket_fetch_and_validate,
            args=(habitat_gs_url, IAM_TOKEN_PATH, habitat_path),
            target_path_list=[habitat_path],
            task_name=f'fetch {habitat_gs_url}')
        fetch_habitat_task_list.append(fetch_habitat_blob_task)

    landmass_bounding_rtree_path = os.path.join(
        WORKING_DIR, _LANDMASS_BOUNDING_RTREE_FILE_PATTERN)

    build_rtree_task = task_graph.add_task(
        func=build_feature_bounding_box_rtree,
        args=(global_polygon_path, landmass_bounding_rtree_path),
        dependent_task_list=[global_polygon_fetch_task],
        target_path_list=[landmass_bounding_rtree_path],
        task_name='simplify_geometry_landmass')

    global_grid_vector_path = os.path.join(
        WORKING_DIR, _GLOBAL_GRID_VECTOR_FILE_PATTERN)

    grid_edges_of_vector_task = task_graph.add_task(
        func=grid_edges_of_vector, args=(
            _GLOBAL_BOUNDING_BOX_WGS84, global_polygon_path,
            landmass_bounding_rtree_path, global_grid_vector_path,
            _WGS84_GRID_SIZE),
        dependent_task_list=[build_rtree_task],
        ignore_path_list=[landmass_bounding_rtree_path],
        target_path_list=[global_grid_vector_path],
        task_name='grid_global_edges')

    grid_edges_of_vector_task.join()

    global_grid_vector = ogr.Open(global_grid_vector_path)
    global_grid_layer = global_grid_vector.GetLayer()
    grid_fid_list = [feature.GetFID() for feature in global_grid_layer]
    global_grid_layer = None
    global_grid_vector = None

    local_rei_point_path_list = []
    wind_exposure_task_list = []
    wave_exposure_task_list = []
    local_fetch_ray_path_list = []
    local_wave_point_path_list = []
    habitat_protection_task_list = []
    local_habitat_protection_path_list = []
    relief_task_list = []
    local_relief_path_list = []
    surge_task_list = []
    local_surge_path_list = []
    for g_index, grid_fid in enumerate(grid_fid_list):
        LOGGER.info("Calculating grid %d of %d", g_index, len(grid_fid_list))
        shore_points_workspace = os.path.join(
            _GRID_WORKSPACES, 'grid_%d' % grid_fid)
        grid_point_path = os.path.join(
            shore_points_workspace, _GRID_POINT_FILE_PATTERN % (grid_fid))

        create_shore_points_task = task_graph.add_task(
            func=create_shore_points, args=(
                global_grid_vector_path, grid_fid, landmass_bounding_rtree_path,
                global_polygon_path, global_wwiii_path, wwiii_rtree_path,
                _SMALLEST_FEATURE_SIZE, shore_points_workspace,
                grid_point_path),
            target_path_list=[grid_point_path],
            ignore_path_list=[
                landmass_bounding_rtree_path, wwiii_rtree_path],
            dependent_task_list=[
                grid_edges_of_vector_task, build_wwiii_task])

        wind_exposure_workspace = os.path.join(
            _WIND_EXPOSURE_WORKSPACES, 'wind_exposure_%d' % grid_fid)
        target_wind_exposure_point_path = os.path.join(
            wind_exposure_workspace,
            _WIND_EXPOSURE_POINT_FILE_PATTERN % grid_fid)
        wind_exposure_task = task_graph.add_task(
            func=calculate_wind_exposure, args=(
                grid_point_path, landmass_bounding_rtree_path,
                global_polygon_path, wind_exposure_workspace,
                _SMALLEST_FEATURE_SIZE, _MAX_FETCH_DISTANCE,
                target_wind_exposure_point_path),
            target_path_list=[target_wind_exposure_point_path],
            ignore_path_list=[landmass_bounding_rtree_path],
            dependent_task_list=[create_shore_points_task])
        wind_exposure_task_list.append(
            wind_exposure_task)
        local_rei_point_path_list.append(
            target_wind_exposure_point_path)
        local_fetch_ray_path_list.append(
            os.path.join(wind_exposure_workspace, 'fetch_rays.gpkg'))

        wave_exposure_workspace = os.path.join(
            _WAVE_EXPOSURE_WORKSPACES, 'wave_exposure_%d' % grid_fid)
        target_wave_exposure_point_path = os.path.join(
            wave_exposure_workspace,
            _WAVE_EXPOSURE_POINT_FILE_PATTERN % grid_fid)
        wave_exposure_task = task_graph.add_task(
            func=calculate_wave_exposure, args=(
                target_wind_exposure_point_path, _MAX_FETCH_DISTANCE,
                wave_exposure_workspace,
                target_wave_exposure_point_path),
            target_path_list=[target_wave_exposure_point_path],
            dependent_task_list=[wind_exposure_task])
        wave_exposure_task_list.append(
            wave_exposure_task)
        local_wave_point_path_list.append(
            target_wave_exposure_point_path)

        habitat_protection_workspace = os.path.join(
            _HABITAT_PROTECTION_WORKSPACES, 'habitat_protection_%d' % grid_fid)
        target_habitat_protection_point_path = os.path.join(
            habitat_protection_workspace,
            _HABITAT_PROTECTION_POINT_FILE_PATTERN % grid_fid)
        habitat_protection_task = task_graph.add_task(
            func=calculate_habitat_protection, args=(
                grid_point_path,
                habitat_vector_lookup,
                habitat_protection_workspace,
                target_habitat_protection_point_path),
            target_path_list=[target_habitat_protection_point_path],
            dependent_task_list=[
                create_shore_points_task] + fetch_habitat_task_list)
        habitat_protection_task_list.append(habitat_protection_task)
        local_habitat_protection_path_list.append(
            target_habitat_protection_point_path)

        relief_workspace = os.path.join(
            _RELIEF_WORKSPACES, 'relief_%d' % grid_fid)
        target_relief_point_path = os.path.join(
            relief_workspace, _RELIEF_POINT_FILE_PATTERN % grid_fid)
        relief_task = task_graph.add_task(
            func=calculate_relief, args=(
                grid_point_path,
                global_dem_path,
                relief_workspace,
                target_relief_point_path),
            target_path_list=[target_relief_point_path],
            dependent_task_list=[
                global_dem_fetch_task, create_shore_points_task])
        relief_task_list.append(relief_task)
        local_relief_path_list.append(
            target_relief_point_path)

        surge_workspace = os.path.join(
            _SURGE_WORKSPACES, 'surge_%d' % grid_fid)
        target_surge_point_vector_path = os.path.join(
            surge_workspace, _SURGE_POINT_FILE_PATTERN % grid_fid)
        surge_task = task_graph.add_task(
            func=calculate_surge, args=(
                grid_point_path,
                global_dem_path,
                surge_workspace,
                target_surge_point_vector_path),
            target_path_list=[target_surge_point_vector_path],
            dependent_task_list=[
                global_dem_fetch_task, create_shore_points_task],
            task_name='calculate surge %d' % grid_fid)
        surge_task_list.append(surge_task)
        local_surge_path_list.append(
            target_surge_point_vector_path)

    target_spatial_reference_wkt = pygeoprocessing.get_vector_info(
        global_polygon_path)['projection']
    merge_vectors_task_list = []
    risk_factor_vector_list = []
    target_habitat_protection_points_path = os.path.join(
        WORKING_DIR, _GLOBAL_HABITAT_PROTECTION_FILE_PATTERN)
    merge_habitat_protection_point_task = task_graph.add_task(
        func=merge_vectors, args=(
            local_habitat_protection_path_list,
            target_spatial_reference_wkt,
            target_habitat_protection_points_path,
            ['Rhab_cur']),
        target_path_list=[target_habitat_protection_points_path],
        dependent_task_list=habitat_protection_task_list)
    risk_factor_vector_list.append(
        (target_habitat_protection_points_path, 'Rhab_cur', 'Rhab_cur'))
    merge_vectors_task_list.append(merge_habitat_protection_point_task)

    target_merged_rei_points_path = os.path.join(
        WORKING_DIR, _GLOBAL_REI_POINT_FILE_PATTERN)
    merge_rei_point_task = task_graph.add_task(
        func=merge_vectors, args=(
            local_rei_point_path_list,
            target_spatial_reference_wkt,
            target_merged_rei_points_path,
            ['REI']),
        target_path_list=[target_merged_rei_points_path],
        dependent_task_list=wind_exposure_task_list)
    merge_vectors_task_list.append(merge_rei_point_task)
    risk_factor_vector_list.append(
        (target_merged_rei_points_path, 'REI', 'Rwind'))

    target_merged_wave_points_path = os.path.join(
        WORKING_DIR, _GLOBAL_WAVE_POINT_FILE_PATTERN)
    merge_wave_point_task = task_graph.add_task(
        func=merge_vectors, args=(
            local_wave_point_path_list,
            target_spatial_reference_wkt,
            target_merged_wave_points_path,
            ['Ew']),
        target_path_list=[target_merged_wave_points_path],
        dependent_task_list=wave_exposure_task_list)
    risk_factor_vector_list.append(
        (target_merged_wave_points_path, 'Ew', 'Rwave'))
    merge_vectors_task_list.append(merge_wave_point_task)

    target_merged_relief_points_path = os.path.join(
        WORKING_DIR, _GLOBAL_RELIEF_POINT_FILE_PATTERN)
    merge_relief_point_task = task_graph.add_task(
        func=merge_vectors, args=(
            local_relief_path_list,
            target_spatial_reference_wkt,
            target_merged_relief_points_path,
            ['relief']),
        target_path_list=[target_merged_relief_points_path],
        dependent_task_list=relief_task_list)
    risk_factor_vector_list.append(
        (target_merged_relief_points_path, 'relief', 'Rrelief'))
    merge_vectors_task_list.append(merge_relief_point_task)

    target_merged_surge_points_path = os.path.join(
        WORKING_DIR, _GLOBAL_SURGE_POINT_FILE_PATTERN)
    merge_surge_task = task_graph.add_task(
        func=merge_vectors, args=(
            local_surge_path_list,
            target_spatial_reference_wkt,
            target_merged_surge_points_path,
            ['surge']),
        target_path_list=[target_merged_surge_points_path],
        dependent_task_list=surge_task_list)
    risk_factor_vector_list.append(
        (target_merged_surge_points_path, 'surge', 'Rsurge'))
    merge_vectors_task_list.append(merge_surge_task)

    target_merged_fetch_rays_path = os.path.join(
        WORKING_DIR, _GLOBAL_FETCH_RAY_FILE_PATTERN)
    target_spatial_reference_wkt = pygeoprocessing.get_vector_info(
        global_polygon_path)['projection']
    _ = task_graph.add_task(
        func=merge_vectors, args=(
            local_fetch_ray_path_list,
            target_spatial_reference_wkt,
            target_merged_fetch_rays_path,
            []),
        target_path_list=[target_merged_fetch_rays_path],
        dependent_task_list=wind_exposure_task_list)

    target_result_point_vector_path = os.path.join(
        WORKING_DIR, _GLOBAL_RISK_RESULT_POINT_VECTOR_FILE_PATTERN)
    summarize_results_task = task_graph.add_task(
        func=summarize_results, args=(
            risk_factor_vector_list, target_result_point_vector_path),
        target_path_list=[target_result_point_vector_path],
        dependent_task_list=merge_vectors_task_list,
        task_name='summarize results')

    # aggregate raster data

    target_raster_aggregate_point_vector_path = os.path.join(
        WORKING_DIR, _AGGREGATE_POINT_VECTOR_FILE_PATTERN)

    aggregate_data_task = task_graph.add_task(
        func=aggregate_raster_data, args=(
            _AGGREGATION_LAYER_MAP, target_result_point_vector_path,
            target_raster_aggregate_point_vector_path),
        target_path_list=[target_raster_aggregate_point_vector_path],
        dependent_task_list=[summarize_results_task],
        task_name='aggregate_raster_data')

    task_graph.close()
    task_graph.join()


def aggregate_raster_data(
        raster_feature_id_map,
        base_point_vector_path,
        target_result_point_vector_path):
    """Add population scenarios and aggregate under each point.

    Parameters:
        raster_feature_id_map (dict): maps feature id names to a 3-tuple:
            * path to a raster to sample for each point in
            `base_point_vector_path
            * boolean indicating whether that value should be divided by the
              pixel area.
            * if not None, a list of raster ids that should be masked to 1
              and everything else to 0. If so result is proportion of 1s in
              sampled area within `sample_distance`
        base_point_vector_path (path): a global point vector path
            that is to be used for the base of the result
        sample_distance (float): distance in meters to sample raster values
            around the point.
        target_result_point_vector_path (path): will contain all the points
            in base_point_vector_path with additional fields mapping
            to the `raster_feature_id_map` keys.

    Returns:
        None.

    """
    if os.path.exists(target_result_point_vector_path):
        os.remove(target_result_point_vector_path)
    base_vector = gdal.OpenEx(base_point_vector_path, gdal.OF_VECTOR)
    mem_result_point_vector = gdal.GetDriverByName('MEMORY').CreateCopy(
        '', base_vector)

    mem_result_point_layer = mem_result_point_vector.GetLayer()
    for simulation_id in raster_feature_id_map:
        mem_result_point_layer.CreateField(
            ogr.FieldDefn(simulation_id, ogr.OFTReal))

    # this is for sea level rise
    mem_result_point_layer.CreateField(
        ogr.FieldDefn('SLRrise_cur', ogr.OFTReal))
    mem_result_point_layer.CreateField(
        ogr.FieldDefn('Rslr_cur', ogr.OFTReal))
    for ssp_id, rcp_id in [(1, 26), (3, 60), (5, 85)]:
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('SLRrise_ssp%d' % ssp_id, ogr.OFTReal))
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('Rhab_ssp%d' % ssp_id, ogr.OFTReal))
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('curpb_ssp%d' % ssp_id, ogr.OFTReal))
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('cpdn_ssp%d' % ssp_id, ogr.OFTReal))
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('Rslr_ssp%d' % ssp_id, ogr.OFTReal))

    # recalibrated population fields for each ssp
    for ssp_id in (1, 3, 5):
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('pdnrc_ssp%d' % ssp_id, ogr.OFTReal))

    for scenario_id in ['cur', 'ssp1', 'ssp3', 'ssp5']:
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('Service_%s' % scenario_id, ogr.OFTReal))
        mem_result_point_layer.CreateField(
            ogr.FieldDefn('NCP_%s' % scenario_id, ogr.OFTReal))

    #UPDATE cv_table SET pdnrc_ssp1 = pdn_gpw * cpdn_ssp1;
    #UPDATE cv_table SET Service_cur = Rtnohab_cur - Rt_cur;
    #UPDATE cv_table SET NCP_cur = Service_cur / Rtnohab_cur;

    for simulation_id, (
                raster_path, divide_by_area, reclass_ids, sample_distance,
                extra_pixel) in (
            raster_feature_id_map.items()):
        raster = gdal.Open(raster_path)
        LOGGER.debug("processing aggregation %s", simulation_id)
        if not raster:
            LOGGER.debug("processing for aggregation %s failed", raster_path)
        band = raster.GetRasterBand(1)
        n_rows = band.YSize
        n_cols = band.XSize
        geotransform = raster.GetGeoTransform()
        nodata = band.GetNoDataValue()

        mem_result_point_layer.ResetReading()
        mem_result_point_layer.StartTransaction()
        for point_feature in mem_result_point_layer:
            point_geometry = point_feature.GetGeometryRef()
            point_x = point_geometry.GetX()
            point_y = point_geometry.GetY()

            lng_m, lat_m = lat_to_meters(point_y)
            pixel_dist_x = int(abs(
                sample_distance / (lng_m * geotransform[1]))) + extra_pixel
            pixel_dist_y = int(abs(
                sample_distance / (lat_m * geotransform[5]))) + extra_pixel
            point_geometry = None

            # this handles the case where slr goes to 360
            if geotransform[0] == 0 and point_x < 0:
                point_x += 360

            pixel_x = int(
                (point_x - geotransform[0]) /
                geotransform[1]) - pixel_dist_x
            pixel_y = int(
                (point_y - geotransform[3]) /
                geotransform[5]) - pixel_dist_y

            if pixel_x < 0:
                pixel_x = 0
            if pixel_y < 0:
                pixel_y = 0
            win_xsize = 1 + pixel_dist_x + extra_pixel
            win_ysize = 1 + pixel_dist_y + extra_pixel
            if pixel_x + win_xsize >= n_cols:
                win_xsize = n_cols - pixel_x - 1
            if pixel_y + win_ysize >= n_rows:
                win_ysize = n_rows - pixel_y - 1
            if win_xsize <= 0 or win_ysize <= 0:
                pixel_value = 0
            else:
                try:
                    array = band.ReadAsArray(
                        xoff=pixel_x, yoff=pixel_y, win_xsize=win_xsize,
                        win_ysize=win_ysize)
                    if nodata is not None and reclass_ids is None:
                        mask = array != nodata
                    else:
                        mask = numpy.ones(array.shape, dtype=numpy.bool)
                    if reclass_ids is not None:
                        pixel_value = numpy.mean(
                            numpy.in1d(array.flatten(), reclass_ids))
                    elif numpy.count_nonzero(mask) > 0:
                        pixel_value = numpy.max(array[mask])
                    else:
                        pixel_value = 0
                except Exception:
                    LOGGER.error(
                        'band size %d %d', band.XSize,
                        band.YSize)
                    raise
            # calculate pixel area in sq km
            if divide_by_area:
                pixel_area_km = abs(
                    (lng_m * geotransform[1]) *
                    (lat_m * geotransform[5])) / 1e6
                pixel_value /= pixel_area_km
            point_feature.SetField(simulation_id, float(pixel_value))
            mem_result_point_layer.SetFeature(point_feature)
        mem_result_point_layer.CommitTransaction()

    mem_result_point_layer.ResetReading()
    mem_result_point_layer.StartTransaction()
    # use this list to calculate risks

    # this list is used to hold all the sea level rise values for cur, ssp1,
    # 3, and 5, hence the length of *4 for the list. it so we can calculate
    # a constant rank for all of them.
    n_points = mem_result_point_layer.GetFeatureCount()
    slr_rise_value_list = numpy.empty(n_points*4)
    fid_index_map = {}
    LOGGER.debug('gathering all the sea level rise data')
    for point_index, point_feature in enumerate(mem_result_point_layer):
        slrrise_val = point_feature.GetField('SLRrate_cur') * 25. / 1000.
        point_feature.SetField('SLRrise_cur', slrrise_val)
        slr_rise_value_list[point_index] = slrrise_val
        fid_index_map[point_index] = point_feature.GetFID()
        for ssp_offset, (ssp_id, rcp_id) in enumerate(
                [(1, 26), (3, 60), (5, 85)]):
            slrrise_val = point_feature.GetField('slr_rcp%d' % rcp_id)
            # this offsets the index by n_points * whatever ssp it is in order
            slr_rise_value_list[point_index + (ssp_offset+1)*n_points] = (
                slrrise_val)
            point_feature.SetField('SLRrise_ssp%d' % ssp_id, slrrise_val)

            point_feature.SetField(
                'Rhab_ssp%d' % ssp_id, (
                    5. - point_feature.GetField('Rhab_cur')) * (
                        point_feature.GetField('urbp_ssp%d' % ssp_id) -
                        point_feature.GetField('urbp_2015')) +
                point_feature.GetField('Rhab_cur'))

            point_feature.SetField(
                'curpb_ssp%d' % ssp_id, (
                    point_feature.GetField('urbp_ssp%d' % ssp_id) -
                    point_feature.GetField('urbp_2015')))

            if point_feature.GetField('pdn_2010') != 0:
                point_feature.SetField(
                    'cpdn_ssp%d' % ssp_id, (
                        point_feature.GetField('pdn_ssp%d' % ssp_id) /
                        point_feature.GetField('pdn_2010')))
            else:
                point_feature.SetField('cpdn_ssp%d' % ssp_id, 0.0)

        #Rt_ssp[1|3|5] = recalculated using Rhab_ssp[1|3|5] and Rslr_ssp[1|3|5]
        #Rt_cur_nh = Rt_cur calculated with Rhab = 5
        #Rt_ssp[1|3|5]_nh = Rt_ssp[1|3|5] calculated with Rhab = 5
        mem_result_point_layer.SetFeature(point_feature)
    LOGGER.debug('calculating Rslr')
    slr_risk_array = numpy.searchsorted(
        numpy.percentile(slr_rise_value_list, [20, 40, 60, 80, 100]),
        slr_rise_value_list) + 1
    for point_index in range(n_points):
        point_feature = mem_result_point_layer.GetFeature(
            fid_index_map[point_index])
        point_feature.SetField('Rslr_cur', float(slr_risk_array[point_index]))
        point_feature.SetField(
            'Rslr_ssp1', float(slr_risk_array[point_index+n_points]))
        point_feature.SetField(
            'Rslr_ssp3', float(slr_risk_array[point_index+n_points*2]))
        point_feature.SetField(
            'Rslr_ssp5', float(slr_risk_array[point_index+n_points*3]))
        mem_result_point_layer.SetFeature(point_feature)
    mem_result_point_layer.CommitTransaction()
    mem_result_point_layer.SyncToDisk()
    mem_result_point_layer = None
    gdal.GetDriverByName('GPKG').CreateCopy(
        target_result_point_vector_path, mem_result_point_vector)

    final_risk_list = [
        (('Rt_cur'), ('Rhab_cur', 'Rslr_cur', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rt_ssp1'), ('Rhab_ssp1', 'Rslr_ssp1', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rt_ssp3'), ('Rhab_ssp3', 'Rslr_ssp3', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rt_ssp5'), ('Rhab_ssp5', 'Rslr_ssp5', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rtnohab_cur'), (5, 'Rslr_cur', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rtnohab_ssp1'), (5, 'Rslr_ssp1', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rtnohab_ssp3'), (5, 'Rslr_ssp3', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge')),
        (('Rtnohab_ssp5'), (5, 'Rslr_ssp5', 'Rwind', 'Rwave', 'Rrelief', 'Rsurge'))]
    calculate_final_risk(final_risk_list, target_result_point_vector_path)


def calculate_final_risk(risk_id_list, target_point_vector_path):
    """Calculate the final Rt and Rtnohabs for cur, ssp1, 3, and 5.

    Parameters:
        base_risk_field_id_list (list): a list of tuples of the form
        (Rt_[cur|ssp{1,3,5}], (list of risk feature ids or constants))
        target_point_vector_path (string): path to vector to modify. Will contain
            at least the fields defined in

            Rt_cur, Rt_ssp1, Rt_ssp3, Rt_ssp5
            Rtnohab_cur, Rtnohab_ssp1, Rtnohab_ssp3, Rtnohab_ssp5

    Returns:
        None.

    """
    target_point_vector = gdal.OpenEx(
        target_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_result_point_layer = target_point_vector.GetLayer()
    for risk_id, _ in risk_id_list:
        target_result_point_layer.CreateField(ogr.FieldDefn(
            risk_id, ogr.OFTReal))
    target_result_point_layer.SyncToDisk()
    target_result_point_layer.ResetReading()
    n_features = target_result_point_layer.GetFeatureCount()
    if n_features > 0:
        LOGGER.debug("calculating final risks")
        target_result_point_layer.StartTransaction()
        for target_feature in target_result_point_layer:
            for risk_id, risk_list in risk_id_list:
                r_list = numpy.array([
                    target_feature.GetField(risk_id)
                    if isinstance(risk_id, str) else risk_id
                    for risk_id in risk_list])
                r_tot = numpy.prod(r_list)**(1./len(r_list))
                target_feature.SetField(risk_id, float(r_tot))

            pdn_gpw = target_feature.GetField('pdn_gpw')
            #UPDATE cv_table SET pdnrc_ssp1 = pdn_gpw * cpdn_ssp1;
            for ssp_id in (1, 3, 5):
                cpdn_ssp = target_feature.GetField('cpdn_ssp%d' % ssp_id)
                if pdn_gpw is not None and cpdn_ssp is not None:
                    target_feature.SetField(
                        'pdnrc_ssp%d' % ssp_id, pdn_gpw*cpdn_ssp)

            #UPDATE cv_table SET Service_cur = Rtnohab_cur - Rt_cur;
            #UPDATE cv_table SET NCP_cur = Service_cur / Rtnohab_cur;
            for scenario_id in ('cur', 'ssp1', 'ssp3', 'ssp5'):
                rtnohab = target_feature.GetField('Rtnohab_%s' % scenario_id)
                rt = target_feature.GetField('Rt_%s' % scenario_id)
                if rtnohab is not None and rt is not None:
                    service = rtnohab - rt
                    target_feature.SetField(
                        'Service_%s' % scenario_id, service)
                    target_feature.SetField(
                        'NCP_%s' % scenario_id, service / rtnohab)

            target_result_point_layer.SetFeature(target_feature)
        target_result_point_layer.CommitTransaction()


def summarize_results(
        risk_factor_vector_list, tm_world_borders_path,
        target_result_point_vector_path):
    """Aggregate the sub factors into a global one.

    This includes sub risk factors into risk factors by percentile.


        risk_factor_vector_list (list): a list of (path, field_id, risk_id)
            global vectors that have at least the fields 'grid_id',
            'point_id', and a `field_id`.  The value `risk_id` is a field
            that will be added to `target_result_point_vector_path` and be
            a risk value between 1 and 5 calculated in most cases from
            the total histogram of the base `field_id` values in the
            point shapefile.
        tm_world_borders_path (str): path to a vector defining country
            borders, has a field called 'name' representing country name.
        target_result_point_vector_path (string): path to target point vector
            with risks and other things I'll write soon.

    Returns:
        None

    """
    if os.path.exists(target_result_point_vector_path):
        os.remove(target_result_point_vector_path)

    countries_myregions_df = pandas.read_csv(
        'countries_myregions_final_md5_7e35a0775335f9aaf9a28adbac0b8895.csv',
        usecols=['country', 'myregions'], sep=',')
    country_to_region_dict = {
        row[1][1]: row[1][0] for row in countries_myregions_df.iterrows()}


    LOGGER.debug("build country spatial index")
    country_rtree, country_geom_fid_map = build_spatial_index(
        tm_world_borders_path)
    country_vector = gdal.OpenEx(tm_world_borders_path, gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()

    base_point_vector_path = risk_factor_vector_list[0][0]
    base_ref_wkt = pygeoprocessing.get_vector_info(
        base_point_vector_path)['projection']
    base_spatial_reference = osr.SpatialReference()
    base_spatial_reference.ImportFromWkt(base_ref_wkt)

    gpkg_driver = ogr.GetDriverByName("GPKG")
    target_result_point_vector = gpkg_driver.CreateDataSource(
        target_result_point_vector_path)
    target_result_point_layer = target_result_point_vector.CreateLayer(
        os.path.splitext(os.path.basename(
            target_result_point_vector_path))[0],
        base_spatial_reference, ogr.wkbPoint)
    risk_id_list = []
    for _, _, risk_id in risk_factor_vector_list:
        target_result_point_layer.CreateField(ogr.FieldDefn(
            risk_id, ogr.OFTReal))
        risk_id_list.append(risk_id)
    target_result_point_layer.CreateField(
        ogr.FieldDefn('country', ogr.OFTString))
    target_result_point_layer.CreateField(
        ogr.FieldDefn('region', ogr.OFTString))
    target_result_point_layer_defn = target_result_point_layer.GetLayerDefn()

    # define initial geometry and fid lookup
    fid_lookup = {}
    risk_factor_vector = ogr.Open(risk_factor_vector_list[0][0])
    risk_factor_layer = risk_factor_vector.GetLayer()
    target_result_point_layer.StartTransaction()
    LOGGER.debug("copying layer")
    for base_point_feature in risk_factor_layer:
        grid_id = base_point_feature.GetField('grid_id')
        point_id = base_point_feature.GetField('point_id')
        fid_lookup[(grid_id, point_id)] = base_point_feature.GetFID()
        target_feature = ogr.Feature(target_result_point_layer_defn)
        target_feature.SetGeometry(
            base_point_feature.GetGeometryRef().Clone())
        point_geom = shapely.wkb.loads(
            target_feature.GetGeometryRef().ExportToWkb())
        # picking 4 because that seems pretty reasonable for nearest countries
        intersection_list = list(country_rtree.nearest(point_geom.bounds, 4))
        min_feature_index = intersection_list[0]
        min_dist = country_geom_fid_map[min_feature_index].distance(
            point_geom)
        for feature_index in intersection_list[1::]:
            dist = country_geom_fid_map[feature_index].distance(point_geom)
            if dist < min_dist:
                min_dist = dist
                min_feature_index = feature_index
        country_name = country_layer.GetFeature(
            min_feature_index).GetField('name')
        target_feature.SetField('country', country_name)
        try:
            target_feature.SetField(
                'region', country_to_region_dict[country_name])
        except KeyError:
            target_feature.SetField('region', 'UNKNOWN')
        target_result_point_layer.CreateFeature(target_feature)

    target_result_point_layer.CommitTransaction()
    target_result_point_layer.SyncToDisk()

    for risk_count, (risk_factor_path, field_id, risk_id) in enumerate(
            risk_factor_vector_list):
        LOGGER.debug(
            "processing risk factor %d of %d %s", risk_count+1,
            target_result_point_vector.GetLayerCount(), risk_factor_path)
        risk_vector = ogr.Open(risk_factor_path)
        risk_layer = risk_vector.GetLayer()
        n_features = target_result_point_layer.GetFeatureCount()
        if n_features == 0:
            continue
        base_risk_values = numpy.empty(n_features)
        fid_index_map = {}
        risk_feature = None
        for feature_index, risk_feature in enumerate(risk_layer):
            risk_value = risk_feature.GetField(field_id)
            point_id = risk_feature.GetField('point_id')
            grid_id = risk_feature.GetField('grid_id')
            target_fid = fid_lookup[(grid_id, point_id)]
            fid_index_map[feature_index] = target_fid
            base_risk_values[feature_index] = float(risk_value)
        # use the last feature to get the grid_id
        if field_id != risk_id:
            # convert to risk
            target_risk_array = numpy.searchsorted(
                numpy.percentile(base_risk_values, [20, 40, 60, 80, 100]),
                base_risk_values) + 1
        else:
            # it's already a risk
            target_risk_array = base_risk_values
        target_result_point_layer.ResetReading()
        target_result_point_layer.StartTransaction()
        for target_index in range(len(fid_index_map)):
            target_fid = fid_index_map[target_index]
            target_feature = target_result_point_layer.GetFeature(
                target_fid)
            target_feature.SetField(
                risk_id, float(target_risk_array[target_index]))
            target_result_point_layer.SetFeature(target_feature)
            target_feature = None
        target_result_point_layer.CommitTransaction()
        target_result_point_layer.SyncToDisk()
    target_result_point_layer = None
    target_result_point_vector = None


def calculate_habitat_protection(
        base_shore_point_vector_path,
        habitat_layer_lookup, workspace_dir,
        target_habitat_protection_point_vector_path):
    """Calculate habitat protection at a set of points.

    Parameters:
        base_shore_point_vector_path (string):  path to a point shapefile to
            analyze habitat protection at.
        habitat_layer_lookup: a dictionary mapping habitat id to a
            (path, rank, distance) tuple
        workspace_dir (string): path to a directory to make local calculations
            in
        target_habitat_protection_point_vector_path (string): path to desired
            output vector.  after completion will have a rank for each
            habitat ID, and a field called Rhab with a value from 1-5
            indicating relative level of protection of that point.

    Returns:
        None.

    """
    try:
        if not os.path.exists(os.path.dirname(
                target_habitat_protection_point_vector_path)):
            os.makedirs(
                os.path.dirname(target_habitat_protection_point_vector_path))
        if os.path.exists(target_habitat_protection_point_vector_path):
            os.remove(target_habitat_protection_point_vector_path)

        base_ref_wkt = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)['projection']
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_ref_wkt)

        # reproject base_shore_point_vector_path to utm coordinates
        base_shore_info = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)
        base_shore_bounding_box = base_shore_info['bounding_box']

        utm_spatial_reference = get_utm_spatial_reference(
            base_shore_info['bounding_box'])
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_shore_info['projection'])

        pygeoprocessing.reproject_vector(
            base_shore_point_vector_path, utm_spatial_reference.ExportToWkt(),
            target_habitat_protection_point_vector_path, driver_name='GPKG')

        utm_bounding_box = pygeoprocessing.get_vector_info(
            target_habitat_protection_point_vector_path)['bounding_box']

        # extend bounding box for max fetch distance
        utm_bounding_box = [
            utm_bounding_box[0] - _MAX_FETCH_DISTANCE,
            utm_bounding_box[1] - _MAX_FETCH_DISTANCE,
            utm_bounding_box[2] + _MAX_FETCH_DISTANCE,
            utm_bounding_box[3] + _MAX_FETCH_DISTANCE]

        # get lat/lng bounding box of utm projected coordinates

        # get global polygon clip of that utm box
        # transform local box back to lat/lng -> global clipping box
        lat_lng_clipping_box = pygeoprocessing.transform_bounding_box(
            utm_bounding_box, utm_spatial_reference.ExportToWkt(),
            base_spatial_reference.ExportToWkt(), edge_samples=11)
        if (base_shore_bounding_box[0] > 0 and
                lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[2] += 360
        elif (base_shore_bounding_box[0] < 0 and
              lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[0] -= 360
        elif base_shore_bounding_box == [0, 0, 0, 0]:
            # this case guards for an empty shore point in case there are
            # very tiny islands or such
            lat_lng_clipping_box = (0, 0, 0, 0)
        lat_lng_clipping_shapely = shapely.geometry.box(*lat_lng_clipping_box)
        habitat_shapely_lookup = {}

        for habitat_id in habitat_layer_lookup:
            habitat_vector = ogr.Open(habitat_layer_lookup[habitat_id][0])
            habitat_layer = habitat_vector.GetLayer()

            # this will hold the clipped landmass geometry
            esri_shapefile_driver = ogr.GetDriverByName("GPKG")
            temp_clipped_vector_path = os.path.join(
                workspace_dir, 'clipped_habitat_%s.gpkg' % habitat_id)
            utm_clipped_vector_path = os.path.join(
                workspace_dir, 'utm_clipped_habitat_%s.gpkg' % habitat_id)
            for path in [temp_clipped_vector_path, utm_clipped_vector_path]:
                if os.path.exists(path):
                    os.remove(path)
            temp_clipped_vector = esri_shapefile_driver.CreateDataSource(
                temp_clipped_vector_path)
            temp_clipped_layer = (
                temp_clipped_vector.CreateLayer(
                    os.path.splitext(temp_clipped_vector_path)[0],
                    base_spatial_reference, ogr.wkbPolygon))
            temp_clipped_defn = temp_clipped_layer.GetLayerDefn()

            # clip global polygon to global clipping box
            for habitat_feature in habitat_layer:
                habitat_shapely = shapely.wkb.loads(
                    habitat_feature.GetGeometryRef().ExportToWkb())
                intersection_shapely = lat_lng_clipping_shapely.intersection(
                    habitat_shapely)
                if intersection_shapely.is_empty:
                    continue
                try:
                    clipped_geometry = ogr.CreateGeometryFromWkt(
                        intersection_shapely.wkt)
                    clipped_feature = ogr.Feature(temp_clipped_defn)
                    clipped_feature.SetGeometry(clipped_geometry)
                    temp_clipped_layer.CreateFeature(clipped_feature)
                    clipped_feature = None
                except Exception:
                    LOGGER.warn(
                        "Couldn't process this intersection %s",
                        intersection_shapely)
            temp_clipped_layer.SyncToDisk()
            temp_clipped_layer = None
            temp_clipped_vector = None

            pygeoprocessing.reproject_vector(
                temp_clipped_vector_path, utm_spatial_reference.ExportToWkt(),
                utm_clipped_vector_path, driver_name='GPKG')

            clipped_geometry_shapely_list = []
            temp_utm_clipped_vector = ogr.Open(utm_clipped_vector_path)
            temp_utm_clipped_layer = temp_utm_clipped_vector.GetLayer()
            for tmp_utm_feature in temp_utm_clipped_layer:
                tmp_utm_geometry = tmp_utm_feature.GetGeometryRef()
                shapely_geometry = shapely.wkb.loads(
                    tmp_utm_geometry.ExportToWkb())
                if shapely_geometry.is_valid:
                    clipped_geometry_shapely_list.append(shapely_geometry)
                tmp_utm_geometry = None
            temp_utm_clipped_layer = None
            temp_utm_clipped_vector = None
            habitat_shapely_lookup[habitat_id] = shapely.ops.cascaded_union(
                clipped_geometry_shapely_list)

        esri_shapefile_driver = ogr.GetDriverByName("GPKG")
        os.path.dirname(base_shore_point_vector_path)
        target_habitat_protection_point_vector = ogr.Open(
            target_habitat_protection_point_vector_path, 1)
        target_habitat_protection_point_layer = (
            target_habitat_protection_point_vector.GetLayer())
        target_habitat_protection_point_layer.CreateField(ogr.FieldDefn(
            'Rhab_cur', ogr.OFTReal))
        for habitat_id in habitat_layer_lookup:
            target_habitat_protection_point_layer.CreateField(ogr.FieldDefn(
                habitat_id, ogr.OFTReal))

        for target_feature in target_habitat_protection_point_layer:
            target_feature_geometry = (
                target_feature.GetGeometryRef().Clone())
            target_feature_shapely = shapely.wkb.loads(
                target_feature_geometry.ExportToWkb())
            min_rank = 5
            sum_sq_rank = 0.0
            for habitat_id, (_, rank, protection_distance) in (
                    habitat_layer_lookup.items()):
                if habitat_shapely_lookup[habitat_id].is_empty:
                    continue
                point_distance_to_feature = target_feature_shapely.distance(
                    habitat_shapely_lookup[habitat_id])
                if point_distance_to_feature <= protection_distance:
                    if rank < min_rank:
                        min_rank = rank
                    sum_sq_rank += (5 - rank) ** 2

            # Equation 4
            if sum_sq_rank > 0:
                r_hab_val = max(
                    1.0, 4.8 - 0.5 * (
                        (1.5 * (5-min_rank))**2 + sum_sq_rank -
                        (5-min_rank)**2)**0.5)
            else:
                r_hab_val = 5.0
            target_feature.SetField('Rhab_cur', r_hab_val)

            target_feature.SetGeometry(target_feature_geometry)
            target_habitat_protection_point_layer.SetFeature(
                target_feature)
        target_habitat_protection_point_layer.SyncToDisk()
        target_habitat_protection_point_layer = None
        target_habitat_protection_point_vector = None
    except Exception:
        traceback.print_exc()
        raise


def calculate_wave_exposure(
        base_fetch_point_vector_path, max_fetch_distance, workspace_dir,
        target_wave_exposure_point_vector_path):
    """Calculate the wave exposure index.

    Parameters:
        base_fetch_point_vector_path (string): path to a point shapefile that
            contains 16 'WavP_[direction]' fields, 'WavPPCT[direction]'
            fields, 'fdist_[direction]' fields, a single H, and a single T
            field.
        max_fetch_distance (float): max fetch distance before a wind fetch ray
            is terminated
        target_wave_exposure_point_vector_path (string): path to an output
            shapefile that will contain a field called 'Ew' which is the
            maximum of ocean or wind waves occurring at that point.

    Returns:
        None

    """
    # this will hold the clipped landmass geometry
    try:
        if not os.path.exists(os.path.dirname(
                target_wave_exposure_point_vector_path)):
            os.makedirs(
                os.path.dirname(target_wave_exposure_point_vector_path))
        if os.path.exists(target_wave_exposure_point_vector_path):
            os.remove(target_wave_exposure_point_vector_path)

        base_ref_wkt = pygeoprocessing.get_vector_info(
            base_fetch_point_vector_path)['projection']
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_ref_wkt)

        esri_shapefile_driver = ogr.GetDriverByName("GPKG")
        os.path.dirname(base_fetch_point_vector_path)
        target_wave_exposure_point_vector = (
            esri_shapefile_driver.CreateDataSource(
                target_wave_exposure_point_vector_path))
        target_wave_exposure_point_layer = (
            target_wave_exposure_point_vector.CreateLayer(
                os.path.splitext(target_wave_exposure_point_vector_path)[0],
                base_spatial_reference, ogr.wkbPoint))
        target_wave_exposure_point_layer.CreateField(ogr.FieldDefn(
            'Ew', ogr.OFTReal))
        target_wave_exposure_point_defn = (
            target_wave_exposure_point_layer.GetLayerDefn())

        base_fetch_point_vector = ogr.Open(base_fetch_point_vector_path)
        base_fetch_point_layer = base_fetch_point_vector.GetLayer()
        for base_fetch_point_feature in base_fetch_point_layer:
            target_feature = ogr.Feature(target_wave_exposure_point_defn)
            target_feature.SetGeometry(
                base_fetch_point_feature.GetGeometryRef().Clone())
            e_local_wedge = (
                0.5 *
                float(base_fetch_point_feature.GetField('H_10PCT'))**2 *
                float(base_fetch_point_feature.GetField('H_10PCT'))) / float(
                    _N_FETCH_RAYS)
            e_ocean = 0.0
            e_local = 0.0
            for sample_index in range(_N_FETCH_RAYS):
                compass_degree = int(sample_index * 360 / 16.)
                fdist = base_fetch_point_feature.GetField(
                    'fdist_%d' % compass_degree)
                if numpy.isclose(fdist, max_fetch_distance):
                    e_ocean += (
                        base_fetch_point_feature.GetField(
                            'WavP_%d' % compass_degree) *
                        base_fetch_point_feature.GetField(
                            'WavPPCT%d' % compass_degree))
                elif fdist > 0.0:
                    e_local += e_local_wedge
            target_feature.SetField('Ew', max(e_ocean, e_local))
            target_wave_exposure_point_layer.CreateFeature(target_feature)
    except Exception:
        traceback.print_exc()
        raise


def calculate_wind_exposure(
        base_shore_point_vector_path,
        landmass_bounding_rtree_path, landmass_vector_path, workspace_dir,
        smallest_feature_size, max_fetch_distance,
        target_fetch_point_vector_path):
    """Calculate wind exposure for each shore point.

    Parameters:
        base_shore_point_vector_path (string): path to a point shapefile
            representing shore points that should be sampled for wind
            exposure.
        landmass_bounding_rtree_path (string): path to an rtree bounding box
            for the landmass polygons.
        landmass_vector_path (string): path to landmass polygon vetor.
        workspace_dir (string): path to a directory that can be created for
            temporary workspace files
        smallest_feature_size (float): smallest feature size to detect in
            meters.
        max_fetch_distance (float): maximum fetch distance for a ray in
            meters.
        target_fetch_point_vector_path (string): path to target point file,
            will be a copy of `base_shore_point_vector_path`'s geometry with
            an 'REI' (relative exposure index) field added.

    Returns:
        None

    """
    try:
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
        os.makedirs(workspace_dir)

        utm_clipped_vector_path = os.path.join(
            workspace_dir, 'utm_clipped_landmass.gpkg')
        temp_fetch_rays_path = os.path.join(
            workspace_dir, 'fetch_rays.gpkg')

        # reproject base_shore_point_vector_path to utm coordinates
        base_shore_info = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)
        base_shore_bounding_box = base_shore_info['bounding_box']

        utm_spatial_reference = get_utm_spatial_reference(
            base_shore_info['bounding_box'])
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_shore_info['projection'])

        pygeoprocessing.reproject_vector(
            base_shore_point_vector_path, utm_spatial_reference.ExportToWkt(),
            target_fetch_point_vector_path, driver_name='GPKG')

        utm_bounding_box = pygeoprocessing.get_vector_info(
            target_fetch_point_vector_path)['bounding_box']

        # extend bounding box for max fetch distance
        utm_bounding_box = [
            utm_bounding_box[0] - max_fetch_distance,
            utm_bounding_box[1] - max_fetch_distance,
            utm_bounding_box[2] + max_fetch_distance,
            utm_bounding_box[3] + max_fetch_distance]

        # get lat/lng bounding box of utm projected coordinates

        # get global polygon clip of that utm box
        # transform local box back to lat/lng -> global clipping box
        lat_lng_clipping_box = pygeoprocessing.transform_bounding_box(
            utm_bounding_box, utm_spatial_reference.ExportToWkt(),
            base_spatial_reference.ExportToWkt(), edge_samples=11)
        if (base_shore_bounding_box[0] > 0 and
                lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[2] += 360
        elif (base_shore_bounding_box[0] < 0 and
              lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[0] -= 360
        elif base_shore_bounding_box == [0, 0, 0, 0]:
            # this case guards for an empty shore point in case there are
            # very tiny islands or such
            lat_lng_clipping_box = (0, 0, 0, 0)
        lat_lng_clipping_shapely = shapely.geometry.box(*lat_lng_clipping_box)

        landmass_vector_rtree = rtree.index.Index(
            os.path.splitext(landmass_bounding_rtree_path)[0])

        landmass_vector = ogr.Open(landmass_vector_path)
        landmass_layer = landmass_vector.GetLayer()

        # this will hold the clipped landmass geometry
        esri_shapefile_driver = ogr.GetDriverByName("GPKG")
        temp_clipped_vector_path = os.path.join(
            workspace_dir, 'clipped_geometry_vector.gpkg')
        temp_clipped_vector = esri_shapefile_driver.CreateDataSource(
            temp_clipped_vector_path)
        temp_clipped_layer = (
            temp_clipped_vector.CreateLayer(
                os.path.splitext(temp_clipped_vector_path)[0],
                base_spatial_reference, ogr.wkbPolygon))
        temp_clipped_defn = temp_clipped_layer.GetLayerDefn()

        # clip global polygon to global clipping box
        for feature_id in landmass_vector_rtree.intersection(
                lat_lng_clipping_box):
            try:
                landmass_feature = landmass_layer.GetFeature(feature_id)
                landmass_shapely = shapely.wkb.loads(
                    landmass_feature.GetGeometryRef().ExportToWkb())
                intersection_shapely = lat_lng_clipping_shapely.intersection(
                    landmass_shapely)
                clipped_geometry = ogr.CreateGeometryFromWkt(
                    intersection_shapely.wkt)
                clipped_feature = ogr.Feature(temp_clipped_defn)
                clipped_feature.SetGeometry(clipped_geometry)
                temp_clipped_layer.CreateFeature(clipped_feature)
                clipped_feature = None
            except Exception:
                clipped_feature = None
                LOGGER.warn(
                    "Couldn't process this intersection %s",
                    intersection_shapely)
        temp_clipped_layer.SyncToDisk()
        temp_clipped_layer = None
        temp_clipped_vector = None

        # project global clipped polygons to UTM
        LOGGER.info("reprojecting grid %s", base_shore_point_vector_path)
        pygeoprocessing.reproject_vector(
            temp_clipped_vector_path, utm_spatial_reference.ExportToWkt(),
            utm_clipped_vector_path, driver_name='GPKG')

        clipped_geometry_shapely_list = []
        temp_utm_clipped_vector = ogr.Open(utm_clipped_vector_path)
        temp_utm_clipped_layer = temp_utm_clipped_vector.GetLayer()
        for tmp_utm_feature in temp_utm_clipped_layer:
            tmp_utm_geometry = tmp_utm_feature.GetGeometryRef()
            shapely_geometry = shapely.wkb.loads(
                tmp_utm_geometry.ExportToWkb())
            if shapely_geometry.is_valid:
                clipped_geometry_shapely_list.append(shapely_geometry)
            tmp_utm_geometry = None
        temp_utm_clipped_layer = None
        temp_utm_clipped_vector = None
        landmass_shapely = shapely.ops.cascaded_union(
            clipped_geometry_shapely_list)
        clipped_geometry_shapely_list = None

        # load land geometry into shapely object
        landmass_shapely_prep = shapely.prepared.prep(landmass_shapely)

        # explode landmass into lines for easy intersection
        temp_polygon_segements_path = os.path.join(
            workspace_dir, 'polygon_segments.gpkg')
        temp_polygon_segments_vector = esri_shapefile_driver.CreateDataSource(
            temp_polygon_segements_path)
        temp_polygon_segments_layer = (
            temp_polygon_segments_vector.CreateLayer(
                os.path.splitext(temp_clipped_vector_path)[0],
                utm_spatial_reference, ogr.wkbLineString))
        temp_polygon_segments_defn = temp_polygon_segments_layer.GetLayerDefn()

        polygon_line_rtree = rtree.index.Index()
        polygon_line_index = []
        shapely_line_index = []
        line_id = 0
        for line in geometry_to_lines(landmass_shapely):
            segment_feature = ogr.Feature(temp_polygon_segments_defn)
            segement_geometry = ogr.Geometry(ogr.wkbLineString)
            segement_geometry.AddPoint(*line.coords[0])
            segement_geometry.AddPoint(*line.coords[1])
            segment_feature.SetGeometry(segement_geometry)
            temp_polygon_segments_layer.CreateFeature(segment_feature)

            if (line.bounds[0] == line.bounds[2] and
                    line.bounds[1] == line.bounds[3]):
                continue
            polygon_line_rtree.insert(line_id, line.bounds)
            line_id += 1
            polygon_line_index.append(segement_geometry)
            shapely_line_index.append(shapely.wkb.loads(
                segement_geometry.ExportToWkb()))

        temp_polygon_segments_layer.SyncToDisk()
        temp_polygon_segments_layer = None
        temp_polygon_segments_vector = None

        # create fetch rays
        temp_fetch_rays_vector = esri_shapefile_driver.CreateDataSource(
            temp_fetch_rays_path)
        temp_fetch_rays_layer = (
            temp_fetch_rays_vector.CreateLayer(
                os.path.splitext(temp_clipped_vector_path)[0],
                utm_spatial_reference, ogr.wkbLineString))
        temp_fetch_rays_defn = temp_fetch_rays_layer.GetLayerDefn()
        temp_fetch_rays_layer.CreateField(ogr.FieldDefn(
            'fetch_dist', ogr.OFTReal))

        target_shore_point_vector = ogr.Open(
            target_fetch_point_vector_path, 1)
        target_shore_point_layer = target_shore_point_vector.GetLayer()
        target_shore_point_layer.CreateField(
            ogr.FieldDefn('REI', ogr.OFTReal))
        for ray_index in range(_N_FETCH_RAYS):
            compass_degree = int(ray_index * 360 / 16.)
            target_shore_point_layer.CreateField(
                ogr.FieldDefn('fdist_%d' % compass_degree, ogr.OFTReal))

        shore_point_logger = _make_logger_callback(
            "Wind exposure %.2f%% complete.", LOGGER)
        # Iterate over every shore point
        for shore_point_feature in target_shore_point_layer:
            shore_point_logger(
                float(shore_point_feature.GetFID()) /
                target_shore_point_layer.GetFeatureCount())
            rei_value = 0.0
            # Iterate over every ray direction
            for sample_index in range(_N_FETCH_RAYS):
                compass_degree = int(sample_index * 360 / 16.)
                compass_theta = float(sample_index) / _N_FETCH_RAYS * 360
                rei_pct = shore_point_feature.GetField(
                    'REI_PCT%d' % int(compass_theta))
                rei_v = shore_point_feature.GetField(
                    'REI_V%d' % int(compass_theta))
                cartesian_theta = -(compass_theta - 90)

                # Determine the direction the ray will point
                delta_x = math.cos(cartesian_theta * math.pi / 180)
                delta_y = math.sin(cartesian_theta * math.pi / 180)

                shore_point_geometry = shore_point_feature.GetGeometryRef()
                point_a_x = (
                    shore_point_geometry.GetX() + delta_x * smallest_feature_size)
                point_a_y = (
                    shore_point_geometry.GetY() + delta_y * smallest_feature_size)
                point_b_x = point_a_x + delta_x * (
                    max_fetch_distance - smallest_feature_size)
                point_b_y = point_a_y + delta_y * (
                    max_fetch_distance - smallest_feature_size)
                shore_point_geometry = None

                # build ray geometry so we can intersect it later
                ray_geometry = ogr.Geometry(ogr.wkbLineString)
                ray_geometry.AddPoint(point_a_x, point_a_y)
                ray_geometry.AddPoint(point_b_x, point_b_y)

                # keep a shapely version of the ray so we can do fast intersection
                # with it and the entire landmass
                ray_point_origin_shapely = shapely.geometry.Point(
                    point_a_x, point_a_y)

                ray_length = 0.0
                if not landmass_shapely_prep.intersects(
                        ray_point_origin_shapely):
                    # the origin is in ocean

                    # This algorithm searches for intersections, if one is found
                    # the ray updates and a smaller intersection set is determined
                    # by experimentation I've found this is significant, but not
                    # an order of magnitude, faster than looping through all
                    # original possible intersections.  Since this algorithm
                    # will be run for a long time, it's worth the additional
                    # complexity
                    tested_indexes = set()
                    while True:
                        intersection = False
                        ray_envelope = ray_geometry.GetEnvelope()
                        for poly_line_index in polygon_line_rtree.intersection(
                                [ray_envelope[i] for i in [0, 2, 1, 3]]):
                            if poly_line_index in tested_indexes:
                                continue
                            tested_indexes.add(poly_line_index)
                            line_segment = (
                                polygon_line_index[poly_line_index])
                            if ray_geometry.Intersects(line_segment):
                                # if the ray intersects the poly line, test if
                                # the intersection is closer than any known
                                # intersection so far
                                intersection_point = ray_geometry.Intersection(
                                    line_segment)
                                # offset the dist with smallest_feature_size
                                # update the endpoint of the ray
                                ray_geometry = ogr.Geometry(ogr.wkbLineString)
                                ray_geometry.AddPoint(point_a_x, point_a_y)
                                ray_geometry.AddPoint(
                                    intersection_point.GetX(),
                                    intersection_point.GetY())
                                intersection = True
                                break
                        if not intersection:
                            break
                    # when we get here `min_point` and `ray_length` are the
                    # minimum intersection points for the ray and the landmass
                    ray_feature = ogr.Feature(temp_fetch_rays_defn)
                    ray_length = ray_geometry.Length()
                    ray_feature.SetField('fetch_dist', ray_length)
                    ray_feature.SetGeometry(ray_geometry)
                    temp_fetch_rays_layer.CreateFeature(ray_feature)
                shore_point_feature.SetField(
                    'fdist_%d' % compass_degree, ray_length)
                ray_feature = None
                ray_geometry = None
                rei_value += ray_length * rei_pct * rei_v
            shore_point_feature.SetField('REI', rei_value)
            target_shore_point_layer.SetFeature(shore_point_feature)

        target_shore_point_layer.SyncToDisk()
        target_shore_point_layer = None
        target_shore_point_vector = None
        temp_fetch_rays_layer.SyncToDisk()
        temp_fetch_rays_layer = None
        temp_fetch_rays_vector = None
    except Exception:
        traceback.print_exc()
        raise


def calculate_relief(
        base_shore_point_vector_path, global_dem_path, workspace_dir,
        target_relief_point_vector_path):
    """Calculate DEM relief as average coastal land area within 5km.

    Parameters:
        base_shore_point_vector_path (string):  path to a point shapefile to
            for relief point analysis.
        global_dem_path (string): path to a DEM raster projected in wgs84.
        workspace_dir (string): path to a directory to make local calculations
            in
        target_relief_point_vector_path (string): path to output vector.
            after completion will a value for average relief within 5km in
            a field called 'relief'.

    Returns:
        None.

    """
    try:
        if not os.path.exists(os.path.dirname(
                target_relief_point_vector_path)):
            os.makedirs(
                os.path.dirname(target_relief_point_vector_path))
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        if os.path.exists(target_relief_point_vector_path):
            os.remove(target_relief_point_vector_path)

        base_ref_wkt = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)['projection']
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_ref_wkt)

        # reproject base_shore_point_vector_path to utm coordinates
        base_shore_info = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)
        base_shore_bounding_box = base_shore_info['bounding_box']

        utm_spatial_reference = get_utm_spatial_reference(
            base_shore_info['bounding_box'])
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_shore_info['projection'])

        pygeoprocessing.reproject_vector(
            base_shore_point_vector_path, utm_spatial_reference.ExportToWkt(),
            target_relief_point_vector_path, driver_name='GPKG')

        utm_bounding_box = pygeoprocessing.get_vector_info(
            target_relief_point_vector_path)['bounding_box']

        target_relief_point_vector = ogr.Open(
            target_relief_point_vector_path, 1)
        target_relief_point_layer = target_relief_point_vector.GetLayer()

        relief_field = ogr.FieldDefn('relief', ogr.OFTReal)
        relief_field.SetPrecision(5)
        relief_field.SetWidth(24)
        target_relief_point_layer.CreateField(relief_field)
        target_relief_point_layer.CreateField(ogr.FieldDefn(
            'id', ogr.OFTInteger))
        for target_feature in target_relief_point_layer:
            target_feature.SetField('id', target_feature.GetFID())
            target_relief_point_layer.SetFeature(target_feature)

        # extend bounding box for max fetch distance
        utm_bounding_box = [
            utm_bounding_box[0] - _MAX_FETCH_DISTANCE,
            utm_bounding_box[1] - _MAX_FETCH_DISTANCE,
            utm_bounding_box[2] + _MAX_FETCH_DISTANCE,
            utm_bounding_box[3] + _MAX_FETCH_DISTANCE]

        # get lat/lng bounding box of utm projected coordinates

        # get global polygon clip of that utm box
        # transform local box back to lat/lng -> global clipping box
        lat_lng_clipping_box = pygeoprocessing.transform_bounding_box(
            utm_bounding_box, utm_spatial_reference.ExportToWkt(),
            base_spatial_reference.ExportToWkt(), edge_samples=11)
        if (base_shore_bounding_box[0] > 0 and
                lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[2] += 360
        elif (base_shore_bounding_box[0] < 0 and
              lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[0] -= 360
        elif base_shore_bounding_box == [0, 0, 0, 0]:
            # this case guards for an empty shore point in case there are
            # very tiny islands or such
            lat_lng_clipping_box = (0, 0, 0, 0)

        clipped_lat_lng_dem_path = os.path.join(
            workspace_dir, 'clipped_lat_lng_dem.tif')

        target_pixel_size = pygeoprocessing.get_raster_info(
            global_dem_path)['pixel_size']
        pygeoprocessing.warp_raster(
            global_dem_path, target_pixel_size, clipped_lat_lng_dem_path,
            'bilinear', target_bb=lat_lng_clipping_box)
        clipped_utm_dem_path = os.path.join(
            workspace_dir, 'clipped_utm_dem.tif')
        target_pixel_size = (
            _SMALLEST_FEATURE_SIZE / 2.0, -_SMALLEST_FEATURE_SIZE / 2.0)
        pygeoprocessing.warp_raster(
            clipped_lat_lng_dem_path, target_pixel_size, clipped_utm_dem_path,
            'bilinear', target_sr_wkt=utm_spatial_reference.ExportToWkt())
        # mask out all DEM < 0 to 0
        nodata = pygeoprocessing.get_raster_info(
            clipped_utm_dem_path)['nodata'][0]

        def zero_negative_values(depth_array):
            valid_mask = depth_array != nodata
            result_array = numpy.empty(
                depth_array.shape, dtype=numpy.int16)
            result_array[:] = nodata
            result_array[valid_mask] = 0
            result_array[depth_array > 0] = depth_array[depth_array > 0]
            return result_array

        positive_dem_path = os.path.join(
            workspace_dir, 'positive_dem.tif')

        pygeoprocessing.raster_calculator(
            [(clipped_utm_dem_path, 1)], zero_negative_values,
            positive_dem_path, gdal.GDT_Int16, nodata)

        # convolve over a 5km radius
        radius_in_pixels = 5000.0 / target_pixel_size[0]
        kernel_filepath = os.path.join(workspace_dir, 'averaging_kernel.tif')
        create_averaging_kernel_raster(radius_in_pixels, kernel_filepath)

        relief_path = os.path.join(workspace_dir, 'relief.tif')
        pygeoprocessing.convolve_2d(
            (positive_dem_path, 1), (kernel_filepath, 1), relief_path)

        relief_raster = gdal.Open(relief_path)
        relief_band = relief_raster.GetRasterBand(1)
        n_rows = relief_band.YSize
        relief_geotransform = relief_raster.GetGeoTransform()
        target_relief_point_layer.ResetReading()
        for point_feature in target_relief_point_layer:
            point_geometry = point_feature.GetGeometryRef()
            point_x, point_y = point_geometry.GetX(), point_geometry.GetY()
            point_geometry = None

            pixel_x = int(
                (point_x - relief_geotransform[0]) / relief_geotransform[1])
            pixel_y = int(
                (point_y - relief_geotransform[3]) / relief_geotransform[5])

            if pixel_y >= n_rows:
                pixel_y = n_rows - 1
            try:
                pixel_value = relief_band.ReadAsArray(
                    xoff=pixel_x, yoff=pixel_y, win_xsize=1,
                    win_ysize=1)[0, 0]
            except Exception:
                LOGGER.error(
                    'relief_band size %d %d', relief_band.XSize,
                    relief_band.YSize)
                raise
            # Make relief "negative" so when we histogram it for risk a
            # "higher" value will show a lower risk.
            point_feature.SetField('relief', -float(pixel_value))
            target_relief_point_layer.SetFeature(point_feature)

        target_relief_point_layer.SyncToDisk()
        target_relief_point_layer = None
        target_relief_point_vector = None

    except Exception:
        traceback.print_exc()
        raise


def calculate_surge(
        base_shore_point_vector_path, global_dem_path, workspace_dir,
        target_surge_point_vector_path):
    """Calculate surge potential as distance to continental shelf (-150m).

    Parameters:
        base_shore_point_vector_path (string):  path to a point shapefile to
            for relief point analysis.
        global_dem_path (string): path to a DEM raster projected in wgs84.
        workspace_dir (string): path to a directory to make local calculations
            in
        target_surge_point_vector_path (string): path to output vector.
            after completion will a value for closest distance to continental
            shelf called 'surge'.

    Returns:
        None.

    """
    try:
        if not os.path.exists(os.path.dirname(
                target_surge_point_vector_path)):
            os.makedirs(
                os.path.dirname(target_surge_point_vector_path))
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        if os.path.exists(target_surge_point_vector_path):
            os.remove(target_surge_point_vector_path)

        base_ref_wkt = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)['projection']
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_ref_wkt)

        # reproject base_shore_point_vector_path to utm coordinates
        base_shore_info = pygeoprocessing.get_vector_info(
            base_shore_point_vector_path)
        base_shore_bounding_box = base_shore_info['bounding_box']

        utm_spatial_reference = get_utm_spatial_reference(
            base_shore_info['bounding_box'])
        base_spatial_reference = osr.SpatialReference()
        base_spatial_reference.ImportFromWkt(base_shore_info['projection'])

        pygeoprocessing.reproject_vector(
            base_shore_point_vector_path, utm_spatial_reference.ExportToWkt(),
            target_surge_point_vector_path, driver_name='GPKG')

        utm_bounding_box = pygeoprocessing.get_vector_info(
            target_surge_point_vector_path)['bounding_box']

        target_relief_point_vector = ogr.Open(
            target_surge_point_vector_path, 1)
        target_relief_point_layer = target_relief_point_vector.GetLayer()

        relief_field = ogr.FieldDefn('surge', ogr.OFTReal)
        target_relief_point_layer.CreateField(relief_field)

        # extend bounding box for max fetch distance
        utm_bounding_box = [
            utm_bounding_box[0] - _MAX_FETCH_DISTANCE,
            utm_bounding_box[1] - _MAX_FETCH_DISTANCE,
            utm_bounding_box[2] + _MAX_FETCH_DISTANCE,
            utm_bounding_box[3] + _MAX_FETCH_DISTANCE]

        # get lat/lng bounding box of utm projected coordinates

        # get global polygon clip of that utm box
        # transform local box back to lat/lng -> global clipping box
        lat_lng_clipping_box = pygeoprocessing.transform_bounding_box(
            utm_bounding_box, utm_spatial_reference.ExportToWkt(),
            base_spatial_reference.ExportToWkt(), edge_samples=11)
        if (base_shore_bounding_box[0] > 0 and
                lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[2] += 360
        elif (base_shore_bounding_box[0] < 0 and
              lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
            lat_lng_clipping_box[0] -= 360
        elif base_shore_bounding_box == [0, 0, 0, 0]:
            # this case guards for an empty shore point in case there are
            # very tiny islands or such
            lat_lng_clipping_box = (0, 0, 0, 0)

        clipped_lat_lng_dem_path = os.path.join(
            workspace_dir, 'clipped_lat_lng_dem.tif')

        target_pixel_size = pygeoprocessing.get_raster_info(
            global_dem_path)['pixel_size']
        pygeoprocessing.warp_raster(
            global_dem_path, target_pixel_size, clipped_lat_lng_dem_path,
            'bilinear', target_bb=lat_lng_clipping_box)
        clipped_utm_dem_path = os.path.join(
            workspace_dir, 'clipped_utm_dem.tif')
        target_pixel_size = (
            _SMALLEST_FEATURE_SIZE / 2.0, -_SMALLEST_FEATURE_SIZE / 2.0)
        pygeoprocessing.warp_raster(
            clipped_lat_lng_dem_path, target_pixel_size, clipped_utm_dem_path,
            'bilinear', target_sr_wkt=utm_spatial_reference.ExportToWkt())
        # mask out all DEM < 0 to 0
        nodata = pygeoprocessing.get_raster_info(
            clipped_utm_dem_path)['nodata'][0]

        shelf_nodata = 2

        def mask_shelf(depth_array):
            valid_mask = depth_array != nodata
            result_array = numpy.empty(
                depth_array.shape, dtype=numpy.int16)
            result_array[:] = shelf_nodata
            result_array[valid_mask] = 0
            result_array[depth_array < -150] = 1
            return result_array

        shelf_mask_path = os.path.join(
            workspace_dir, 'shelf_mask.tif')

        pygeoprocessing.raster_calculator(
            [(clipped_utm_dem_path, 1)], mask_shelf,
            shelf_mask_path, gdal.GDT_Byte, shelf_nodata)

        # convolve to find edges
        # grid shoreline from raster
        shelf_kernel_path = os.path.join(workspace_dir, 'shelf_kernel.tif')
        shelf_convoultion_raster_path = os.path.join(
            workspace_dir, 'shelf_convolution.tif')
        make_shore_kernel(shelf_kernel_path)
        pygeoprocessing.convolve_2d(
            (shelf_mask_path, 1), (shelf_kernel_path, 1),
            shelf_convoultion_raster_path, target_datatype=gdal.GDT_Byte,
            target_nodata=255)

        nodata = pygeoprocessing.get_raster_info(
            shelf_convoultion_raster_path)['nodata'][0]

        def _shelf_mask_op(shelf_convolution):
            """Mask values on land that border the continental shelf."""
            result = numpy.empty(shelf_convolution.shape, dtype=numpy.uint8)
            result[:] = nodata
            valid_mask = shelf_convolution != nodata
            # If a pixel is on land, it gets at least a 9, but if it's all on
            # land it gets an 17 (8 neighboring pixels), so we search between 9
            # and 17 to determine a shore pixel
            result[valid_mask] = numpy.where(
                (shelf_convolution[valid_mask] >= 9) &
                (shelf_convolution[valid_mask] < 17), 1, nodata)
            return result

        shelf_edge_raster_path = os.path.join(workspace_dir, 'shelf_edge.tif')
        pygeoprocessing.raster_calculator(
            [(shelf_convoultion_raster_path, 1)], _shelf_mask_op,
            shelf_edge_raster_path, gdal.GDT_Byte, nodata)

        shore_geotransform = pygeoprocessing.get_raster_info(
            shelf_edge_raster_path)['geotransform']

        shelf_rtree = rtree.index.Index()

        for offset_info, data_block in pygeoprocessing.iterblocks(
                (shelf_edge_raster_path, 1)):
            row_indexes, col_indexes = numpy.mgrid[
                offset_info['yoff']:offset_info['yoff']+offset_info['win_ysize'],
                offset_info['xoff']:offset_info['xoff']+offset_info['win_xsize']]
            valid_mask = data_block == 1
            x_coordinates = (
                shore_geotransform[0] +
                shore_geotransform[1] * (col_indexes[valid_mask] + 0.5) +
                shore_geotransform[2] * (row_indexes[valid_mask] + 0.5))
            y_coordinates = (
                shore_geotransform[3] +
                shore_geotransform[4] * (col_indexes[valid_mask] + 0.5) +
                shore_geotransform[5] * (row_indexes[valid_mask] + 0.5))

            for x_coord, y_coord in zip(x_coordinates, y_coordinates):
                shelf_rtree.insert(
                    0, [x_coord, y_coord, x_coord, y_coord],
                    obj=shapely.geometry.Point(x_coord, y_coord))

        for point_feature in target_relief_point_layer:
            point_geometry = point_feature.GetGeometryRef()
            point_shapely = shapely.wkb.loads(point_geometry.ExportToWkb())
            nearest_point = list(shelf_rtree.nearest(
                    (point_geometry.GetX(),
                     point_geometry.GetY(),
                     point_geometry.GetX(),
                     point_geometry.GetY()),
                    objects='raw', num_results=1))
            if len(nearest_point) > 0:
                distance = nearest_point[0].distance(point_shapely)
                point_feature.SetField('surge', float(distance))
            else:
                # so far away it's essentially not an issue
                point_feature.SetField('surge', 0.0)
            target_relief_point_layer.SetFeature(point_feature)

        target_relief_point_layer.SyncToDisk()
        target_relief_point_layer = None
        target_relief_point_vector = None

    except Exception:
        traceback.print_exc()
        LOGGER.exception('exception in calculating surge')
        raise


def create_averaging_kernel_raster(radius_in_pixels, kernel_filepath):
    """Create a raster kernel with a radius given.

    Parameters:
        expected_distance (int or float): The distance (in pixels) of the
            kernel's radius, the distance at which the value of the decay
            function is equal to `1/e`.
        kernel_filepath (string): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.

    Returns:
        None

    """
    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_filepath.encode('utf-8'), int(radius_in_pixels)*2+1,
        int(radius_in_pixels)*2+1,
        1, gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS('NAD27')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    integration = 0.0
    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = numpy.indices((row_block_width,
                                                      col_block_width),
                                                     dtype=numpy.float)

            row_indices += numpy.float(row_offset - radius_in_pixels)
            col_indices += numpy.float(col_offset - radius_in_pixels)

            kernel_index_distances = numpy.hypot(
                row_indices, col_indices)
            kernel = numpy.where(
                kernel_index_distances > radius_in_pixels, 0.0, 1.0)
            integration += numpy.sum(kernel)

            kernel_band.WriteArray(kernel, xoff=col_offset,
                                   yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_dataset.FlushCache()

    for block_data, kernel_block in pygeoprocessing.iterblocks(
            (kernel_filepath, 1)):
        kernel_block /= integration
        kernel_band.WriteArray(kernel_block, xoff=block_data['xoff'],
                               yoff=block_data['yoff'])


def create_shore_points(
        sample_grid_vector_path, grid_fid, landmass_bounding_rtree_path,
        landmass_vector_path, wwiii_vector_path, wwiii_rtree_path,
        smallest_feature_size,
        workspace_dir, target_shore_point_vector_path):
    """Create points that lie on the coast line of the landmass.

    Parameters:
        sample_grid_vector_path (string): path to vector containing grids
            that are used for discrete global sampling of the landmass
            polygon.
        grid_fid (integer): feature ID in `sample_grid_vector_path`'s layer to
            operate on.
        landmass_bounding_rtree_path (string): path to an rtree index that has
            bounding box indexes of the polygons in `landmass_vector_path`.
        landmass_vector_path (string): path to polygon vector representing
            landmass.
        wwiii_vector_path (string): path to point shapefile representing
            the Wave Watch III data.
        wwiii_rtree_path (string): path to an rtree index that has
            the points of `wwiii_vector_path` indexed.
        smallest_feature_size (float): smallest feature size to grid a shore
            point on.
        workspace_dir (string): path to a directory that can be created
            during run to hold temporary files.  Will be deleted on successful
            function completion.
        target_shore_point_vector_path (string): path to a point vector that
            will be created and contain points on the shore of the landmass.

    Returns:
        None.

    """
    LOGGER.info("Creating shore points for grid %s", grid_fid)
    # create the spatial reference from the base vector
    landmass_spatial_reference = osr.SpatialReference()
    landmass_spatial_reference.ImportFromWkt(
        pygeoprocessing.get_vector_info(landmass_vector_path)['projection'])

    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)

    lat_lng_clipped_vector_path = os.path.join(
        workspace_dir, 'clipped_geometry_lat_lng.gpkg')
    grid_raster_path = os.path.join(workspace_dir, 'grid.tif')
    convolution_raster_path = os.path.join(
        workspace_dir, 'convolution.tif')
    utm_clipped_vector_path = os.path.join(
        workspace_dir, 'clipped_geometry_utm.gpkg')
    shore_kernel_path = os.path.join(
        workspace_dir, 'shore_kernel.tif')
    shore_raster_path = os.path.join(
        workspace_dir, 'shore_raster.tif')

    for path in [target_shore_point_vector_path,
                 lat_lng_clipped_vector_path,
                 grid_raster_path]:
        if os.path.exists(path):
            os.remove(path)

    esri_shapefile_driver = ogr.GetDriverByName("GPKG")

    # this will hold the clipped landmass geometry
    lat_lng_clipped_vector = esri_shapefile_driver.CreateDataSource(
        lat_lng_clipped_vector_path)
    lat_lng_clipped_layer = (
        lat_lng_clipped_vector.CreateLayer(
            os.path.splitext(lat_lng_clipped_vector_path)[0],
            landmass_spatial_reference, ogr.wkbPolygon))
    lat_lng_clipped_defn = lat_lng_clipped_layer.GetLayerDefn()

    # this will hold the output sample points on the shore
    target_shore_point_vector = esri_shapefile_driver.CreateDataSource(
        target_shore_point_vector_path)
    target_shore_point_layer = target_shore_point_vector.CreateLayer(
        os.path.splitext(target_shore_point_vector_path)[0],
        landmass_spatial_reference, ogr.wkbPoint)

    wwiii_vector = ogr.Open(wwiii_vector_path)
    wwiii_layer = wwiii_vector.GetLayer()
    wwiii_defn = wwiii_layer.GetLayerDefn()
    field_names = []
    for field_index in range(wwiii_defn.GetFieldCount()):
        field_defn = wwiii_defn.GetFieldDefn(field_index)
        field_name = field_defn.GetName()
        if field_name in ['I', 'J']:
            continue
        field_names.append(field_name)
        target_shore_point_layer.CreateField(field_defn)
    target_shore_point_defn = target_shore_point_layer.GetLayerDefn()

    landmass_vector = ogr.Open(landmass_vector_path)
    landmass_layer = landmass_vector.GetLayer()

    grid_vector = ogr.Open(sample_grid_vector_path)
    grid_layer = grid_vector.GetLayer()
    grid_feature = grid_layer.GetFeature(grid_fid)
    grid_geometry_ref = grid_feature.GetGeometryRef()
    grid_shapely = shapely.wkb.loads(grid_geometry_ref.ExportToWkb())

    landmass_vector_rtree = rtree.index.Index(
        os.path.splitext(landmass_bounding_rtree_path)[0])

    # project global polygon clip to UTM
    # transform lat/lng box to utm -> local box
    utm_spatial_reference = get_utm_spatial_reference(grid_shapely.bounds)
    utm_bounding_box = pygeoprocessing.transform_bounding_box(
        grid_shapely.bounds, landmass_spatial_reference.ExportToWkt(),
        utm_spatial_reference.ExportToWkt(), edge_samples=11)

    # add a pixel buffer so we clip land that's a little outside the grid
    pixel_buffer = 1
    utm_bounding_box[0] -= pixel_buffer * smallest_feature_size
    utm_bounding_box[1] -= pixel_buffer * smallest_feature_size
    utm_bounding_box[2] += pixel_buffer * smallest_feature_size
    utm_bounding_box[3] += pixel_buffer * smallest_feature_size

    # transform local box back to lat/lng -> global clipping box
    lat_lng_clipping_box = pygeoprocessing.transform_bounding_box(
        utm_bounding_box, utm_spatial_reference.ExportToWkt(),
        landmass_spatial_reference.ExportToWkt(), edge_samples=11)
    # see if we're wrapped on the dateline
    if (grid_shapely.bounds[0] > 0 and
            lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
        lat_lng_clipping_box[2] += 360
    elif (grid_shapely.bounds[0] < 0 and
          lat_lng_clipping_box[0] > lat_lng_clipping_box[2]):
        lat_lng_clipping_box[0] -= 360
    lat_lng_clipping_shapely = shapely.geometry.box(*lat_lng_clipping_box)

    # clip global polygon to utm clipping box
    LOGGER.info(
        "clip global polygon to utm clipping box for grid %s", grid_fid)
    for feature_id in landmass_vector_rtree.intersection(
            lat_lng_clipping_box):
        try:
            base_feature = landmass_layer.GetFeature(feature_id)
            base_geometry = base_feature.GetGeometryRef()
            base_shapely = shapely.wkb.loads(base_geometry.ExportToWkb())
            base_geometry = None
            intersection_shapely = lat_lng_clipping_shapely.intersection(
                base_shapely)
            target_geometry = ogr.CreateGeometryFromWkt(
                intersection_shapely.wkt)
            target_feature = ogr.Feature(lat_lng_clipped_defn)
            target_feature.SetGeometry(target_geometry)
            lat_lng_clipped_layer.CreateFeature(target_feature)
            target_feature = None
            target_geometry = None
        except shapely.errors.WKBReadingError:
            LOGGER.error("couldn't read fid %d for some reason.", feature_id)
        except Exception:
            LOGGER.warn(
                "Couldn't process this intersection %s",
                intersection_shapely)
    lat_lng_clipped_layer.SyncToDisk()
    lat_lng_clipped_layer = None
    lat_lng_clipped_vector = None

    # create grid for underlying local utm box
    pygeoprocessing.reproject_vector(
        lat_lng_clipped_vector_path, utm_spatial_reference.ExportToWkt(),
        utm_clipped_vector_path, driver_name='GPKG')

    byte_nodata = 255

    pygeoprocessing.create_raster_from_vector_extents(
        utm_clipped_vector_path,
        grid_raster_path, (
            smallest_feature_size / 2.0, -smallest_feature_size / 2.0),
        gdal.GDT_Byte, byte_nodata, fill_value=0)

    # rasterize utm global clip to grid
    pygeoprocessing.rasterize(
        utm_clipped_vector_path, grid_raster_path, [1], None)

    # grid shoreline from raster
    make_shore_kernel(shore_kernel_path)
    pygeoprocessing.convolve_2d(
        (grid_raster_path, 1), (shore_kernel_path, 1),
        convolution_raster_path, target_datatype=gdal.GDT_Byte,
        target_nodata=255)

    temp_grid_nodata = pygeoprocessing.get_raster_info(
        grid_raster_path)['nodata'][0]

    def _shore_mask_op(shore_convolution):
        """Mask values on land that border water."""
        result = numpy.empty(shore_convolution.shape, dtype=numpy.uint8)
        result[:] = byte_nodata
        valid_mask = shore_convolution != temp_grid_nodata
        # If a pixel is on land, it gets at least a 9, but if it's all on
        # land it gets an 17 (8 neighboring pixels), so we search between 9
        # and 17 to determine a shore pixel
        result[valid_mask] = numpy.where(
            (shore_convolution[valid_mask] >= 9) &
            (shore_convolution[valid_mask] < 17), 1, byte_nodata)
        return result

    pygeoprocessing.raster_calculator(
        [(convolution_raster_path, 1)], _shore_mask_op,
        shore_raster_path, gdal.GDT_Byte, byte_nodata)

    shore_geotransform = pygeoprocessing.get_raster_info(
        shore_raster_path)['geotransform']

    utm_to_base_transform = osr.CoordinateTransformation(
        utm_spatial_reference, landmass_spatial_reference)

    # rtree index loads without the extension
    wwiii_rtree_base_path = os.path.splitext(
        wwiii_rtree_path)[0]
    wwiii_rtree = rtree.index.Index(wwiii_rtree_base_path)
    wwiii_field_lookup = {}

    LOGGER.info(
        "Interpolating shore points with Wave Watch III data for grid %s",
        grid_fid)
    feature_lookup = {}
    for offset_info, data_block in pygeoprocessing.iterblocks(
            (shore_raster_path, 1)):
        row_indexes, col_indexes = numpy.mgrid[
            offset_info['yoff']:offset_info['yoff']+offset_info['win_ysize'],
            offset_info['xoff']:offset_info['xoff']+offset_info['win_xsize']]
        valid_mask = data_block == 1
        x_coordinates = (
            shore_geotransform[0] +
            shore_geotransform[1] * (col_indexes[valid_mask] + 0.5) +
            shore_geotransform[2] * (row_indexes[valid_mask] + 0.5))
        y_coordinates = (
            shore_geotransform[3] +
            shore_geotransform[4] * (col_indexes[valid_mask] + 0.5) +
            shore_geotransform[5] * (row_indexes[valid_mask] + 0.5))

        for x_coord, y_coord in zip(x_coordinates, y_coordinates):
            shore_point_geometry = ogr.Geometry(ogr.wkbPoint)
            shore_point_geometry.AddPoint(x_coord, y_coord)
            shore_point_geometry.Transform(utm_to_base_transform)
            # make sure shore point is within the bounding box of the gri
            if grid_geometry_ref.Contains(shore_point_geometry):
                shore_point_feature = ogr.Feature(target_shore_point_defn)
                shore_point_feature.SetGeometry(shore_point_geometry)

                # get the nearest wave watch III points from the shore point
                nearest_points = list(wwiii_rtree.nearest(
                    (shore_point_geometry.GetX(),
                     shore_point_geometry.GetY(),
                     shore_point_geometry.GetX(),
                     shore_point_geometry.GetY()), 3))[0:3]

                # create placeholders for point geometry and field values
                wwiii_points = numpy.empty((3, 2))
                wwiii_values = numpy.empty((3, len(field_names)))
                for fid_index, fid in enumerate(nearest_points):
                    wwiii_feature = wwiii_layer.GetFeature(fid)
                    wwiii_geometry = wwiii_feature.GetGeometryRef()
                    wwiii_points[fid_index] = numpy.array(
                        [wwiii_geometry.GetX(), wwiii_geometry.GetY()])
                    try:
                        wwiii_values[fid_index] = wwiii_field_lookup[fid]
                    except KeyError:
                        wwiii_field_lookup[fid] = numpy.array(
                            [float(wwiii_feature.GetField(field_name))
                             for field_name in field_names])
                        wwiii_values[fid_index] = wwiii_field_lookup[fid]

                distance = numpy.linalg.norm(
                    wwiii_points - numpy.array(
                        (shore_point_geometry.GetX(),
                         shore_point_geometry.GetY())))

                # make sure we're within a valid data distance
                if distance > _MAX_WWIII_DISTANCE:
                    continue

                wwiii_values *= distance
                wwiii_values = numpy.mean(wwiii_values, axis=0)
                wwiii_values /= numpy.sum(distance)

                for field_name_index, field_name in enumerate(field_names):
                    shore_point_feature.SetField(
                        field_name, wwiii_values[field_name_index])

                target_shore_point_layer.CreateFeature(shore_point_feature)
                shore_point_feature = None
    del feature_lookup
    LOGGER.info("All done with shore points for grid %s", grid_fid)


def grid_edges_of_vector(
        base_bounding_box, base_vector_path,
        base_feature_bounding_box_rtree_path, target_grid_vector_path,
        target_grid_size):
    """Build a sparse grid covering the edges of the base polygons.

    Parameters:
        base_bounding_box (list/tuple): format [minx, miny, maxx, maxy]
            represents the bounding box of the underlying grid cells to be
            created.
        base_vector_path (string): path to shapefile of polygon features,
            a grid cell will be created if it contains the *edge* of any
            feature.
        base_feature_bounding_box_rtree_path (string): path to an rtree that
            indexes the bounding boxes of the polygon features in
            `base_vector_path`.
        target_grid_vector_path (string): path to shapefile grid that will be
            created by this function.
        target_grid_size (float): length of side of the grid cell to be
            created in the target_grid_vector.
        done_token_path (string): path to a file to create when the function
            is complete.

    Returns:
        None

    """
    LOGGER.info("Building global grid.")
    n_rows = int((
        base_bounding_box[3]-base_bounding_box[1]) / float(
            target_grid_size))
    n_cols = int((
        base_bounding_box[2]-base_bounding_box[0]) / float(
            target_grid_size))

    if os.path.exists(target_grid_vector_path):
        os.remove(target_grid_vector_path)

    # create the target vector as an GPKG
    esri_shapefile_driver = ogr.GetDriverByName("GPKG")
    target_grid_vector = esri_shapefile_driver.CreateDataSource(
        target_grid_vector_path)

    # create the spatial reference from the base vector
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(
        pygeoprocessing.get_vector_info(base_vector_path)['projection'])

    target_grid_layer = target_grid_vector.CreateLayer(
        os.path.splitext(target_grid_vector_path)[0],
        spatial_reference, ogr.wkbPolygon)

    target_grid_defn = target_grid_layer.GetLayerDefn()

    base_vector = ogr.Open(base_vector_path)
    base_layer = base_vector.GetLayer()

    # the input path has a .dat extension, but the `rtree` package only uses
    # the basename.  It's a quirk of the library, so we'll deal with it by
    # cutting off the extension.
    target_feature_rtree_index = rtree.index.Index(
        os.path.splitext(base_feature_bounding_box_rtree_path)[0])

    logger_callback = _make_logger_callback(
        'Cell coverage %.2f%% complete', LOGGER)

    prepared_geometry = {}
    for cell_index in range(n_rows * n_cols):
        logger_callback(float(cell_index) / (n_rows * n_cols))
        row_index = cell_index / n_cols
        col_index = cell_index % n_cols
        # format of bounding box is  [xmin, ymin, xmax, ymax]
        cell_bounding_box = [
            base_bounding_box[0]+col_index*target_grid_size,
            base_bounding_box[1]+row_index*target_grid_size,
            base_bounding_box[0]+(col_index+1)*target_grid_size,
            base_bounding_box[1]+(row_index+1)*target_grid_size]

        intersections = list(
            target_feature_rtree_index.intersection(cell_bounding_box))
        if len(intersections) == 0:
            # skip this cell if no intersections with the bounding boxes occur
            continue

        # construct cell geometry both in OGR and Shapely formats
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i, j in [(0, 1), (2, 1), (2, 3), (0, 3), (0, 1)]:
            ring.AddPoint(cell_bounding_box[i], cell_bounding_box[j])
        cell_geometry = ogr.Geometry(ogr.wkbPolygon)
        cell_geometry.AddGeometry(ring)
        cell_geometry_shapely = shapely.wkb.loads(
            cell_geometry.ExportToWkb())

        for fid in intersections:
            if fid not in prepared_geometry:
                base_feature = base_layer.GetFeature(fid)
                base_feature_geometry = base_feature.GetGeometryRef()
                prepared_geometry[fid] = shapely.wkb.loads(
                    base_feature_geometry.ExportToWkb())
                base_feature_geometry = None

            if (prepared_geometry[fid].intersects(
                    cell_geometry_shapely) and
                    not prepared_geometry[fid].contains(
                        cell_geometry_shapely)):
                # add cell to target layer if it intersects the edge of a
                # base polygon feature
                target_feature = ogr.Feature(target_grid_defn)
                target_feature.SetGeometry(cell_geometry)
                target_grid_layer.CreateFeature(target_feature)
                target_feature = None
                # no need to test the rest if one intersects
                break
    target_grid_layer.SyncToDisk()
    target_grid_layer = None
    target_grid_vector = None


def get_utm_spatial_reference(bounding_box):
    """Determine UTM spatial reference given lat/lng bounding box.

    Parameter:
        bounding_box (list/tuple): UTM84 coordinate bounding box in the
            format [min_lng, min_lat, max_lng, max_lat]

    Returns:
        An osr.SpatialReference that corresponds to the UTM zone in the
            median point of the bounding box.

    """
    # project lulc_map to UTM zone median
    mean_long = (bounding_box[0] + bounding_box[2]) / 2
    mean_lat = (bounding_box[1] + bounding_box[3]) / 2
    utm_code = (math.floor((mean_long + 180)/6) % 60) + 1

    # Determine if north or sourth
    lat_code = 6 if mean_lat > 0 else 7

    # and this is the format of an EPSG UTM code:
    epsg_code = int('32%d%02d' % (lat_code, utm_code))

    utm_sr = osr.SpatialReference()
    utm_sr.ImportFromEPSG(epsg_code)
    return utm_sr


def build_feature_bounding_box_rtree(vector_path, target_rtree_path):
    """Build an r-tree index of the global feature envelopes.

    Parameter:
        vector_path (string): path to vector to build bounding box index for
        target_rtree_path (string): path to ".dat" file to store the saved
            r-tree.  A ValueError is raised if this file already exists

    Returns:
        None.

    """
    # the input path has a .dat extension, but the `rtree` package only uses
    # the basename.  It's a quirk of the library, so we'll deal with it by
    # cutting off the extension.
    global_feature_index_base = os.path.splitext(
        target_rtree_path)[0]
    LOGGER.info("Building rtree index at %s", global_feature_index_base)
    if os.path.exists(target_rtree_path):
        for ext in ['.dat', '.idx']:
            os.remove(global_feature_index_base + ext)
    global_feature_index = rtree.index.Index(global_feature_index_base)

    global_vector = ogr.Open(vector_path)
    global_layer = global_vector.GetLayer()
    n_features = global_layer.GetFeatureCount()

    logger_callback = _make_logger_callback(
        'rTree construction %.2f%% complete', LOGGER)

    for feature_index, global_feature in enumerate(global_layer):
        feature_geometry = global_feature.GetGeometryRef()
        # format of envelope is [minx, maxx, miny, maxy]
        feature_envelope = feature_geometry.GetEnvelope()
        # format of tree bounding box is [minx, miny, maxx, maxy]
        global_feature_index.insert(
            global_feature.GetFID(), (
                feature_envelope[0], feature_envelope[2],
                feature_envelope[1], feature_envelope[3]))
        logger_callback(float(feature_index) / n_features)
    global_feature_index.close()


def make_shore_kernel(kernel_path):
    """Make a 3x3 raster with a 9 in the middle and 1s on the outside."""
    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_path.encode('utf-8'), 3, 3, 1,
        gdal.GDT_Byte)

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_raster.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(127)
    kernel_band.WriteArray(numpy.array([[1, 1, 1], [1, 9, 1], [1, 1, 1]]))


def _make_logger_callback(message, logger):
    """Build a timed logger callback that prints `message` replaced.

    Parameters:
        message (string): a string that expects a %f replacement variable for
            `proportion_complete`.

    Returns:
        Function with signature:
            logger_callback(proportion_complete, psz_message, p_progress_arg)

    """
    def logger_callback(proportion_complete):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (proportion_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                LOGGER.info(message, proportion_complete * 100)
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0

    return logger_callback


def build_wwiii_rtree(wwiii_vector_path, wwiii_rtree_path):
    """Build RTree indexed by FID for points in `wwwiii_vector_path`."""
    base_wwiii_rtree_path = os.path.splitext(wwiii_rtree_path)[0]
    if os.path.exists(wwiii_rtree_path):
        for ext in ['.dat', '.idx']:
            os.remove(base_wwiii_rtree_path+ext)
    wwiii_rtree = rtree.index.Index(base_wwiii_rtree_path)

    wwiii_vector = ogr.Open(wwiii_vector_path)
    wwiii_layer = wwiii_vector.GetLayer()
    for wwiii_feature in wwiii_layer:
        wwiii_geometry = wwiii_feature.GetGeometryRef()
        wwiii_x = wwiii_geometry.GetX()
        wwiii_y = wwiii_geometry.GetY()
        wwiii_rtree.insert(
            wwiii_feature.GetFID(), (wwiii_x, wwiii_y, wwiii_x, wwiii_y))
    wwiii_layer = None
    wwiii_vector = None


def merge_vectors(
        base_vector_path_list, target_spatial_reference_wkt,
        target_merged_vector_path, field_list_to_copy):
    """Merge all the vectors in the `base_vector_path_list`.

    Parameters:
        base_vector_path_list (list): a list of OGR DataSources.  Should
            all be single layer identical feature definitions.
        target_merged_vector_path (string): path to desired output vector.
            will contain a single layer with all merged features from the
            base vectors.
        field_list_to_copy (string): list of field names to copy from the base
            vectors to the target vectors

    Returns:
        None

    """
    target_spatial_reference = osr.SpatialReference()
    target_spatial_reference.ImportFromWkt(target_spatial_reference_wkt)

    gpkg_driver = ogr.GetDriverByName('GPKG')
    if os.path.exists(target_merged_vector_path):
        gpkg_driver.DeleteDataSource(target_merged_vector_path)
    base_vector = ogr.Open(base_vector_path_list[0])
    base_layer = base_vector.GetLayer()
    base_layer_defn = base_layer.GetLayerDefn()

    target_vector = gpkg_driver.CreateDataSource(target_merged_vector_path)
    target_layer = target_vector.CreateLayer(
        target_merged_vector_path, srs=target_spatial_reference,
        geom_type=base_layer.GetGeomType())
    for field_name in field_list_to_copy:
        target_layer.CreateField(
            base_layer_defn.GetFieldDefn(
                base_layer_defn.GetFieldIndex(field_name)))
    target_layer.CreateField(ogr.FieldDefn('grid_id', ogr.OFTInteger))
    target_layer.CreateField(ogr.FieldDefn('point_id', ogr.OFTInteger))

    base_layer = None
    base_vector = None

    for base_vector_path in base_vector_path_list:
        base_vector = ogr.Open(base_vector_path)
        base_layer = base_vector.GetLayer()
        grid_id = int(
            re.search('.*_(\d+)', os.path.split(
                os.path.dirname(base_vector_path))[1]).group(1))

        base_spatial_reference = base_layer.GetSpatialRef()
        base_to_target_transform = osr.CoordinateTransformation(
            base_spatial_reference, target_spatial_reference)

        for feature in base_layer:
            target_feat = ogr.Feature(target_layer.GetLayerDefn())
            target_geometry = feature.GetGeometryRef().Clone()
            target_geometry.Transform(base_to_target_transform)
            target_feat.SetGeometry(target_geometry)
            for field_name in field_list_to_copy:
                target_feat.SetField(field_name, feature.GetField(field_name))
            target_feat.SetField('point_id', feature.GetFID())
            target_feat.SetField('grid_id', grid_id)
            target_layer.CreateFeature(target_feat)
            target_feat = None
    target_layer.SyncToDisk()
    target_layer = None
    target_vector = None


def geometry_to_lines(geometry):
    """Convert a geometry object to a list of lines."""
    if geometry.type == 'Polygon':
        return polygon_to_lines(geometry)
    elif geometry.type == 'MultiPolygon':
        line_list = []
        for geom in geometry.geoms:
            line_list.extend(geometry_to_lines(geom))
        return line_list
    else:
        return []


def polygon_to_lines(geometry):
    """Return a list of shapely lines given higher order shapely geometry."""
    line_list = []
    last_point = geometry.exterior.coords[0]
    for point in geometry.exterior.coords[1::]:
        if point == last_point:
            continue
        line_list.append(shapely.geometry.LineString([last_point, point]))
        last_point = point
    line_list.append(shapely.geometry.LineString([
        last_point, geometry.exterior.coords[0]]))
    for interior in geometry.interiors:
        last_point = interior.coords[0]
        for point in interior.coords[1::]:
            if point == last_point:
                continue
            line_list.append(shapely.geometry.LineString([last_point, point]))
            last_point = point
        line_list.append(shapely.geometry.LineString([
            last_point, interior.coords[0]]))
    return line_list


def lat_to_meters(lat):
    """Return (lng, lat) in meters."""
    m1 = 111132.92
    m2 = -559.82
    m3 = 1.175
    m4 = -0.0023
    p1 = 111412.84
    p2 = -93.5
    p3 = 0.118

    lat = lat * math.pi / 180

    latlen = (
        m1 + (m2 * math.cos(2 * lat)) + (m3 * math.cos(4 * lat)) + (m4 * math.cos(6 * lat)))
    longlen = abs(
        (p1 * math.cos(lat)) + (p2 * math.cos(3 * lat)) + (p3 * math.cos(5 * lat)))

    return (longlen, latlen)


def unzip_file(zipfile_path, target_dir, touchfile_path):
    """Unzip contents of `zipfile_path`.

    Parameters:
        zipfile_path (string): path to a zipped file.
        target_dir (string): path to extract zip file to.
        touchfile_path (string): path to a file to create if unzipping is
            successful.

    Returns:
        None.

    """
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    with open(touchfile_path, 'w') as touchfile:
        touchfile.write(f'unzipped {zipfile_path}')


def google_bucket_fetch_and_validate(url, json_key_path, target_path):
    """Create a function to download a Google Blob to a given path.

    Parameters:
        url (string): url to blob, matches the form
            '^https://storage.cloud.google.com/([^/]*)/(.*)$'
        json_key_path (string): path to Google Cloud private key generated by
        https://cloud.google.com/iam/docs/creating-managing-service-account-keys
        target_path (string): path to target file.

    Returns:
        a function with a single `path` argument to the target file. Invoking
            this function will download the Blob to `path`.

    """
    url_matcher = re.match(
        r'^https://[^/]*\.com/([^/]*)/(.*)$', url)
    LOGGER.debug(url)
    client = google.cloud.storage.client.Client.from_service_account_json(
        json_key_path)
    bucket_id = url_matcher.group(1)
    LOGGER.debug(f'parsing bucket {bucket_id} from {url}')
    bucket = client.get_bucket(bucket_id)
    blob_id = url_matcher.group(2)
    LOGGER.debug(f'loading blob {blob_id} from {url}')
    blob = google.cloud.storage.Blob(
        blob_id, bucket, chunk_size=2**24)
    LOGGER.info(f'downloading blob {target_path} from {url}')
    try:
        os.makedirs(os.path.dirname(target_path))
    except os.error:
        pass
    blob.download_to_filename(target_path)
    if not reproduce.valid_hash(target_path, 'embedded'):
        raise ValueError(f"{target_path}' does not match its expected hash")


def build_spatial_index(vector_path):
    """Build an rtree/geom list tuple from ``vector_path``."""
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    geom_index = rtree.index.Index()
    geom_fid_dict = {}
    for feature in layer:
        geom = feature.GetGeometryRef()
        shapely_geom = shapely.wkb.loads(geom.ExportToWkb())
        geom_fid_dict[feature.GetFID()] = shapely_geom
        geom_index.insert(feature.GetFID(), shapely_geom.bounds)
    return geom_index, geom_fid_dict


def threshold_raster_op(
        base_raster_path, min_val, max_val, target_raster_path):
    """Threshold base raster to 1.0 if between min & max val."""

    def threshold_op(val):
        return (val >= min_val) & (val <= max_val)

    pygeoprocessing.raster_calculator(
        ((base_raster_path, 1),), threshold_op, target_raster_path,
        gdal.GDT_Byte, 2)


def mult2_op(array_a, array_b):
    """Multiply two arrays together blindly."""
    return array_a * array_b


def fetch_validate_and_unzip(
        gs_path, iam_token_path, download_dir, target_path):
    """Fetch a gzipped file, validate it, and unzip to `target_path`."""
    target_gz_path = os.path.join(download_dir, os.path.basename(gs_path))
    reproduce.utils.google_bucket_fetch_and_validate(
        gs_path, iam_token_path, target_gz_path)

    with gzip.open(target_gz_path, 'rb') as gzip_file:
        with open(target_path, 'wb') as target_file:
            shutil.copyfileobj(gzip_file, target_file)


if __name__ == '__main__':
    main()
