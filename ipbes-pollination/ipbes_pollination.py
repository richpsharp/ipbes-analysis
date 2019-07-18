"""
Pollination analysis for IPBES.

    From "IPBES Methods: Pollination Contribution to Human Nutrition."
    https://www.dropbox.com/s/gc4b1miw2zypuke/IPBES%20Methods_Pollination_RS.docx?dl=0
"""
import subprocess
import shutil
import base64
import sys
import zipfile
import time
import os
import re
import logging
import itertools
import multiprocessing
import tempfile

import rtree
import shapely.wkb
import shapely.prepared
from osgeo import gdal
from osgeo import ogr
import google.cloud.client
import google.cloud.storage
from osgeo import osr
import pandas
import numpy
import scipy.ndimage.morphology
import reproduce
import taskgraph
import pygeoprocessing

HG_REV = subprocess.check_output(['hg', 'id', '--id']).decode('utf-8').strip()
if HG_REV.endswith('+'):
    raise RuntimeError(
        "Current repository has uncommitted changes. Commit them.")
HG_DATE = subprocess.check_output(
    ['hg', 'log', '-l', '1', '--template', '{date|isodate}']).decode(
        'utf-8').strip().replace(' ', '_')
HG_SUFFIX = f'hg_{HG_DATE}_{HG_REV}'

# set a 1GB limit for the cache
gdal.SetCacheMax(2**30)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger('ipbes_pollination')
_MULT_NODATA = -1
# the following are the globio landcover codes. A tuple (x, y) indicates the
# inclusive range from x to y. Pollinator habitat was defined as any natural
# land covers, as defined (GLOBIO land-cover classes 6, secondary vegetation,
# and  50-180, various types of primary vegetation). To test sensitivity to
# this definition we included "semi-natural" habitats (GLOBIO land-cover
# classes 3, 4, and 5; pasture, rangeland and forestry, respectively) in
# addition to "natural", and repeated all analyses with semi-natural  plus
# natural habitats, but this did not substantially alter the results  so we do
# not include it in our final analysis or code base.

GLOBIO_AG_CODES = [2, (10, 40), (230, 232)]
GLOBIO_NATURAL_CODES = [6, (50, 180)]

WORKING_DIR = './ipbes_pollination_workspace'
OUTPUT_DIR = os.path.join(WORKING_DIR, 'outputs')
ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

try:
    GOOGLE_BUCKET_KEY_PATH = os.path.normpath(sys.argv[1])
except IndexError:
    raise RuntimeError("Expected command line argument of path to bucket key")

NODATA = -9999
N_WORKERS = -1#max(1, multiprocessing.cpu_count())

GOOGLE_BUCKET_ID = 'ipbes-pollination-result'
BLOB_ROOT = f'''ipbes_pollination_result'''


def main():
    """Entry point."""
    try:
        os.makedirs(WORKING_DIR)
    except OSError:
        pass

    task_graph = taskgraph.TaskGraph(
        CHURN_DIR, N_WORKERS, reporting_interval=5.0)

    summary_raster_path_map = {}

    # 1.2.    POLLINATION-DEPENDENT NUTRIENT PRODUCTION
    # Pollination-dependence of crops, crop yields, and crop micronutrient
    # content were combined in an analysis to calculate pollination-dependent
    # nutrient production, following Chaplin-Kramer et al. (2012).
    # 1.2.2.  Pollination dependency
    # Crop pollination dependency was determined for 115 crops (permanent link
    # to table). Dependency was defined as the percent by which yields are
    # reduced for each crop with inadequate pollination (ranging from 0-95%),
    # according to Klein et al. (2007).

    # Crop content of critical macro and micronutrients (KJ energy/100 g, IU
    #   Vitamin A/ 100 g and mcg Folate/100g) for the 115 crops were taken
    #   from USDA (2011) . The USDA (2011) data also provided estimated refuse
    #   of the food item (e.g., peels, seeds). The pollination-dependent yield
    #   was reduced by this refuse percent and then multiplied by nutrient
    #   content, and summed across all crops to derive pollination-dependent
    #   nutrient yields (KJ/ha, IU Vitamin A/ha, mcg Folate/ha) for each
    #   nutrient at 5 arc min. The full table used in this analysis can be
    # found at https://storage.googleapis.com/ecoshard-root/'
    # 'crop_nutrient_md5_d6e67fd79ef95ab2dd44ca3432e9bb4d.csv
    crop_nutrient_url = (
        'https://storage.googleapis.com/ecoshard-root/'
        'crop_nutrient_md5_2fbe7455357f8008a12827fd88816fc1.csv')
    crop_nutrient_table_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(crop_nutrient_url))

    crop_nutrient_table_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            crop_nutrient_url, GOOGLE_BUCKET_KEY_PATH,
            crop_nutrient_table_path),
        target_path_list=[crop_nutrient_table_path],
        task_name=f'fetch {os.path.basename(crop_nutrient_table_path)}')

    degree_basedata_url = (
        'https://storage.cloud.google.com/ecoshard-root/ipbes/'
        'degree_basedata_md5_73a03fa0f5fb622e8d0f07c616576677.zip')
    degree_zipfile_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(degree_basedata_url))
    degree_basedata_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            degree_basedata_url, GOOGLE_BUCKET_KEY_PATH,
            degree_zipfile_path),
        target_path_list=[degree_zipfile_path],
        task_name=f'fetch {os.path.basename(degree_zipfile_path)}')
    zip_touch_file_path = os.path.join(
        os.path.dirname(degree_zipfile_path), 'degree_basedata_zip.txt')
    __ = task_graph.add_task(
        func=unzip_file,
        args=(
            degree_zipfile_path, os.path.dirname(degree_zipfile_path),
            zip_touch_file_path),
        target_path_list=[zip_touch_file_path],
        dependent_task_list=[degree_basedata_fetch_task],
        task_name=f'unzip degree_basedata_zip')

    tm_world_borders_basedata_url = (
        'https://storage.cloud.google.com/ecoshard-root/ipbes/'
        'TM_WORLD_BORDERS_SIMPL-0.3_md5_15057f7b17752048f9bd2e2e607fe99c.zip')
    tm_world_borders_zipfile_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(tm_world_borders_basedata_url))
    tm_world_borders_basedata_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            tm_world_borders_basedata_url, GOOGLE_BUCKET_KEY_PATH,
            tm_world_borders_zipfile_path),
        target_path_list=[tm_world_borders_zipfile_path],
        task_name=f'fetch {os.path.basename(tm_world_borders_zipfile_path)}')
    zip_touch_file_path = os.path.join(
        os.path.dirname(tm_world_borders_zipfile_path),
        'tm_world_borders_basedata_zip.txt')
    __ = task_graph.add_task(
        func=unzip_file,
        args=(
            tm_world_borders_zipfile_path, os.path.dirname(
                tm_world_borders_zipfile_path),
            zip_touch_file_path),
        target_path_list=[zip_touch_file_path],
        dependent_task_list=[tm_world_borders_basedata_fetch_task],
        task_name=f'unzip tm_world_borders_basedata_zip')

    hunger_basedata_url = (
        'https://storage.cloud.google.com/ecoshard-root/ipbes/'
        'hunger-shapefile_md5_9f2b0e3d07d2002cd97db22e2a9b9069.zip')
    hunger_zipfile_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(hunger_basedata_url))
    hunger_basedata_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            hunger_basedata_url, GOOGLE_BUCKET_KEY_PATH,
            hunger_zipfile_path),
        target_path_list=[hunger_zipfile_path],
        task_name=f'fetch {os.path.basename(hunger_zipfile_path)}')
    zip_touch_file_path = os.path.join(
        os.path.dirname(hunger_zipfile_path),
        'hunger_basedata_zip.txt')
    __ = task_graph.add_task(
        func=unzip_file,
        args=(
            hunger_zipfile_path, os.path.dirname(
                hunger_zipfile_path),
            zip_touch_file_path),
        target_path_list=[zip_touch_file_path],
        dependent_task_list=[hunger_basedata_fetch_task],
        task_name=f'unzip hunger_basedata_zip')

    landcover_data = {
        'GLOBIO4_LU_10sec_2050_SSP5_RCP85': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/GLOBIO4_LU_10sec_2050_SSP5_RCP85_'
            'md5_1b3cc1ce6d0ff14d66da676ef194f130.tif', 'ssp5'),
        'GLOBIO4_LU_10sec_2050_SSP1_RCP26': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/GLOBIO4_LU_10sec_2050_SSP1_RCP26_md5_'
            '803166420f51e5ef7dcaa970faa98173.tif', 'ssp1'),
        'GLOBIO4_LU_10sec_2050_SSP3_RCP70': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/GLOBIO4_LU_10sec_2050_SSP3_RCP70_md5_'
            'e77077a3220a36f7f0441bbd0f7f14ab.tif', 'ssp3'),
        'Globio4_landuse_10sec_1850': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/Globio4_landuse_10sec_1850_md5_'
            '0b7fcb4b180d46b4fc2245beee76d6b9.tif', '1850'),
        'Globio4_landuse_10sec_2015': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/Globio4_landuse_10sec_2015_md5_'
            '939a57c2437cd09bd5a9eb472b9bd781.tif', 'cur'),
        'Globio4_landuse_10sec_1980': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/Globio4_landuse_10sec_1980_md5_'
            'f6384eac7579318524439df9530ca1f4.tif', '1980'),
        'Globio4_landuse_10sec_1945': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/Globio4_landuse_10sec_1945_md5_'
            '52c7b4c38c26defefa61132fd25c5584.tif', '1945'),
        'Globio4_landuse_10sec_1910': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/Globio4_landuse_10sec_1910_md5_'
            'e7da8fa29db305ff63c99fed7ca8d5e2.tif', '1910'),
        'Globio4_landuse_10sec_1900': (
            'https://storage.cloud.google.com/ecoshard-root/globio_landcover'
            '/Globio4_landuse_10sec_1900_md5_'
            'f5db818a5b16799bf2cb627e574120a4.tif', '1900'),
        'ESACCI_LC_L4_LCSS': ('https://storage.cloud.google.com/ipbes-ndr-ecoshard-data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7_md5_1254d25f937e6d9bdee5779d377c5aa4.tif', 'ESACCI_LC_L4_LCSS')
        }

    # 1.2.3.  Crop production

    # Spatially-explicit global crop yields (tons/ha) at 5 arc min (~10 km)
    # were taken from Monfreda et al. (2008) for 115 crops (permanent link to
    # crop yield folder). These yields were multiplied by crop pollination
    # dependency to calculate the pollination-dependent crop yield for each 5
    # min grid cell. Note the monfreda maps are in units of per-hectare
    # yields

    yield_and_harea_zip_url = (
        'https://storage.cloud.google.com/ecoshard-root/ipbes/'
        'monfreda_2008_observed_yield_and_harea_'
        'md5_49b529f57071cc85abbd02b6e105089b.zip')

    yield_and_harea_zip_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(yield_and_harea_zip_url))

    yield_zip_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            yield_and_harea_zip_url, GOOGLE_BUCKET_KEY_PATH,
            yield_and_harea_zip_path),
        target_path_list=[yield_and_harea_zip_path],
        task_name=f'fetch {os.path.basename(yield_and_harea_zip_path)}')

    zip_touch_file_path = os.path.join(
        os.path.dirname(yield_and_harea_zip_path),
        'monfreda_2008_observed_yield_and_harea.txt')
    unzip_yield_task = task_graph.add_task(
        func=unzip_file,
        args=(
            yield_and_harea_zip_path,
            os.path.dirname(yield_and_harea_zip_path), zip_touch_file_path),
        target_path_list=[zip_touch_file_path],
        dependent_task_list=[yield_zip_fetch_task],
        task_name=f'unzip monfreda_2008_observed_yield')

    # fetch a landcover map to use as a base for the dimensions of the
    # production raster, later on we'll fetch them all.
    landcover_key, (landcover_url, _) = next(iter(landcover_data.items()))
    landcover_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(landcover_url))

    landcover_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(landcover_url, GOOGLE_BUCKET_KEY_PATH, landcover_path),
        target_path_list=[landcover_path],
        task_name=f'fetch {landcover_key}')

    yield_and_harea_raster_dir = os.path.join(
        os.path.dirname(yield_and_harea_zip_path),
        'monfreda_2008_observed_yield_and_harea')

    prod_total_nut_10s_task_path_map = {}
    poll_dep_prod_nut_10s_task_path_map = {}
    for nut_id, nutrient_name in [
            ('en', 'Energy'), ('va', 'VitA'), ('fo', 'Folate')]:
        # total annual production of nutrient
        yield_total_nut_10km_path = os.path.join(
            CHURN_DIR, 'monfreda_2008_yield_nutrient_rasters',
            f'monfreda_2008_yield_total_{nut_id}_10km.tif')
        yield_total_nut_10s_path = os.path.join(
            CHURN_DIR, 'monfreda_2008_yield_nutrient_rasters',
            f'monfreda_2008_yield_total_{nut_id}_10s.tif')
        prod_total_nut_10s_path = os.path.join(
            CHURN_DIR, 'monfreda_2008_prod_nutrient_rasters',
            f'monfreda_2008_prod_total_{nut_id}_10s.tif')

        prod_total_task = task_graph.add_task(
            func=create_prod_nutrient_raster,
            args=(
                 crop_nutrient_table_path, nutrient_name,
                 yield_and_harea_raster_dir, False, landcover_path,
                 yield_total_nut_10km_path, yield_total_nut_10s_path,
                 prod_total_nut_10s_path),
            target_path_list=[
                yield_total_nut_10km_path, yield_total_nut_10s_path,
                prod_total_nut_10s_path],
            dependent_task_list=[
                landcover_fetch_task, unzip_yield_task,
                crop_nutrient_table_fetch_task],
            task_name=f"""create prod raster {
                os.path.basename(prod_total_nut_10s_path)}""")
        prod_total_nut_10s_task_path_map[nut_id] = (
            prod_total_task, prod_total_nut_10s_path)

        # pollination-dependent annual production of nutrient
        poll_dep_yield_nut_10km_path = os.path.join(
            CHURN_DIR, 'monfreda_2008_yield_poll_dep_rasters',
            f'monfreda_2008_yield_poll_dep_{nut_id}_10km.tif')
        poll_dep_yield_nut_10s_path = os.path.join(
            CHURN_DIR, 'monfreda_2008_yield_poll_dep_rasters',
            f'monfreda_2008_yield_poll_dep_{nut_id}_10s.tif')
        poll_dep_prod_nut_10s_path = os.path.join(
            CHURN_DIR, 'monfreda_2008_prod_poll_dep_rasters',
            f'monfreda_2008_prod_poll_dep_{nut_id}_10s.tif')
        pol_dep_prod_task = task_graph.add_task(
            func=create_prod_nutrient_raster,
            args=(
                crop_nutrient_table_path, nutrient_name,
                yield_and_harea_raster_dir, True, landcover_path,
                poll_dep_yield_nut_10km_path, poll_dep_yield_nut_10s_path,
                poll_dep_prod_nut_10s_path),
            target_path_list=[
                poll_dep_yield_nut_10km_path, poll_dep_yield_nut_10s_path,
                poll_dep_prod_nut_10s_path],
            dependent_task_list=[landcover_fetch_task, unzip_yield_task],
            task_name=f"""create poll dep production raster {
                os.path.basename(poll_dep_prod_nut_10s_path)}""")
        poll_dep_prod_nut_10s_task_path_map[nut_id] = (
            pol_dep_prod_task, poll_dep_prod_nut_10s_path)

    # The proportional area of natural within 2 km was calculated for every
    #  pixel of agricultural land (GLOBIO land-cover classes 2, 230, 231, and
    #  232) at 10 arc seconds (~300 m) resolution. This 2 km scale represents
    #  the distance most commonly found to be predictive of pollination
    #  services (Kennedy et al. 2013).
    kernel_raster_path = os.path.join(CHURN_DIR, 'radial_kernel.tif')
    kernel_task = task_graph.add_task(
        func=create_radial_convolution_mask,
        args=(0.00277778, 2000., kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name='make convolution kernel')

    prod_poll_dep_realized_1d_task_path_map = {}
    prod_poll_dep_unrealized_1d_task_path_map = {}
    prod_total_realized_1d_task_path_map = {}
    # mask landcover into agriculture and pollinator habitat
    for landcover_key, (landcover_url, landcover_short_suffix) in (
            landcover_data.items()):
        landcover_path = os.path.join(
            ECOSHARD_DIR, os.path.basename(landcover_url))
        landcover_fetch_task = task_graph.add_task(
            func=google_bucket_fetch_and_validate,
            args=(landcover_url, GOOGLE_BUCKET_KEY_PATH, landcover_path),
            target_path_list=[landcover_path],
            task_name=f'fetch {landcover_key}')

        # This loop is so we don't duplicate code for 'ag' and 'hab' with the
        # only difference being the lulc codes and prefix
        for mask_prefix, globio_codes in [
                ('ag', GLOBIO_AG_CODES), ('hab', GLOBIO_NATURAL_CODES)]:
            mask_key = f'{landcover_key}_{mask_prefix}_mask'
            mask_target_path = os.path.join(
                CHURN_DIR, f'{mask_prefix}_mask', f'{mask_key}.tif')

            mask_task = task_graph.add_task(
                func=mask_raster,
                args=(landcover_path, globio_codes, mask_target_path),
                target_path_list=[mask_target_path],
                dependent_task_list=[landcover_fetch_task],
                task_name=f'mask {mask_key}',)

            if mask_prefix == 'hab':
                hab_task_path_tuple = (mask_task, mask_target_path)
            elif mask_prefix == 'ag':
                ag_task_path_tuple = (mask_task, mask_target_path)

        one_degree_task, mask_count_nonzero_1d_path = (
            schedule_aggregate_to_degree(
                task_graph, ag_task_path_tuple[1], numpy.count_nonzero,
                ag_task_path_tuple[0],
                target_raster_path_name=os.path.join(
                    OUTPUT_DIR,
                    f'''{landcover_key}_ag_mask_count_nonzero_1d.tif''')))
        summary_raster_path_map[f'''{
            landcover_key}_ag_mask_count_nonzero_1d'''] = (
                mask_count_nonzero_1d_path)

        pollhab_2km_prop_path = os.path.join(
            CHURN_DIR, 'pollhab_2km_prop',
            f'pollhab_2km_prop_{landcover_key}.tif')
        pollhab_2km_prop_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=[
                (hab_task_path_tuple[1], 1), (kernel_raster_path, 1),
                pollhab_2km_prop_path],
            kwargs={
                'working_dir': CHURN_DIR,
                'ignore_nodata': True,
                'gtiff_creation_options': (
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
                    'PREDICTOR=3', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
                    'NUM_THREADS=2'),
                'n_threads': 4},
            dependent_task_list=[hab_task_path_tuple[0], kernel_task],
            target_path_list=[pollhab_2km_prop_path],
            task_name=(
                'calculate proportional'
                f' {os.path.basename(pollhab_2km_prop_path)}'))

        # calculate pollhab_2km_prop_on_ag_10s by multiplying pollhab_2km_prop
        # by the ag mask
        pollhab_2km_prop_on_ag_path = os.path.join(
            OUTPUT_DIR, f'''pollhab_2km_prop_on_ag_10s_{
                landcover_short_suffix}.tif''')
        pollhab_2km_prop_on_ag_task = task_graph.add_task(
            func=mult_rasters,
            args=(
                ag_task_path_tuple[1], pollhab_2km_prop_path,
                pollhab_2km_prop_on_ag_path),
            target_path_list=[pollhab_2km_prop_on_ag_path],
            dependent_task_list=[
                pollhab_2km_prop_task, ag_task_path_tuple[0]],
            task_name=(
                f'''pollhab 2km prop on ag {
                    os.path.basename(pollhab_2km_prop_on_ag_path)}'''))

        #  1.1.4.  Sufficiency threshold A threshold of 0.3 was set to
        #  evaluate whether there was sufficient pollinator habitat in the 2
        #  km around farmland to provide pollination services, based on
        #  Kremen et al.'s (2005)  estimate of the area requirements for
        #  achieving full pollination. This produced a map of wild
        #  pollination sufficiency where every agricultural pixel was
        #  designated in a binary fashion: 0 if proportional area of habitat
        #  was less than 0.3; 1 if greater than 0.3. Maps of pollination
        #  sufficiency can be found at (permanent link to output), outputs
        #  "poll_suff_..." below.

        threshold_val = 0.3
        pollinator_suff_hab_path = os.path.join(
            CHURN_DIR, 'poll_suff_hab_ag_coverage_rasters',
            f'poll_suff_ag_coverage_prop_10s_{landcover_short_suffix}.tif')
        poll_suff_task = task_graph.add_task(
            func=threshold_select_raster,
            args=(
                pollhab_2km_prop_path,
                ag_task_path_tuple[1], threshold_val,
                pollinator_suff_hab_path),
            target_path_list=[pollinator_suff_hab_path],
            dependent_task_list=[
                pollhab_2km_prop_task, ag_task_path_tuple[0]],
            task_name=f"""poll_suff_ag_coverage_prop {
                os.path.basename(pollinator_suff_hab_path)}""")

        one_degree_task, poll_suff_ag_coverage_sum_1d_path = (
            schedule_aggregate_to_degree(
                task_graph, pollinator_suff_hab_path, numpy.sum,
                poll_suff_task,
                target_raster_path_name=os.path.join(
                    OUTPUT_DIR, f'''poll_suff_ag_coverage_prop_sum_1d_{
                        landcover_short_suffix}.tif''')))
        summary_raster_path_map[f'''poll_suff_ag_coverage_prop_sum_1d_{
            landcover_short_suffix}'''] = poll_suff_ag_coverage_sum_1d_path

        one_degree_task, poll_suff_ag_coverage_count_nonzero_1d_path = (
            schedule_aggregate_to_degree(
                task_graph, pollinator_suff_hab_path, numpy.count_nonzero,
                poll_suff_task,
                target_raster_path_name=os.path.join(
                    OUTPUT_DIR,
                    f'''poll_suff_ag_coverage_prop_count_nonzero_1d_{
                        landcover_short_suffix}.tif''')))
        summary_raster_path_map[
            f'''poll_suff_ag_coverage_prop_count_nonzero_1d_{
                landcover_short_suffix}'''] = (
                    poll_suff_ag_coverage_count_nonzero_1d_path)

        one_degree_task, poll_suff_ag_coverage_count_nonzero_1d_path = (
            schedule_aggregate_to_degree(
                task_graph, pollinator_suff_hab_path, count_ge_one,
                poll_suff_task,
                target_raster_path_name=os.path.join(
                    OUTPUT_DIR,
                    f'''poll_suff_ag_coverage_prop_count_eq1_1d_{
                        landcover_short_suffix}.tif''')))
        summary_raster_path_map[
            f'''poll_suff_ag_coverage_prop_count_eq1_1d_{
                landcover_short_suffix}'''] = (
                    poll_suff_ag_coverage_count_nonzero_1d_path)

        # tot_prod_en|va|fo_10s|1d_cur|ssp1|ssp3|ssp5
        # total annual production of energy (KJ/yr), vitamin A (IU/yr),
        # and folate (mg/yr)
        nat_cont_task_path_map = {}
        poll_cont_prod_map = {}
        poll_cont_1d_prod_map = {}
        for nut_id in ('en', 'va', 'fo'):
            tot_prod_task, tot_prod_path = (
                prod_total_nut_10s_task_path_map[nut_id])

            prod_total_potential_path = os.path.join(
                OUTPUT_DIR, f'''prod_total_potential_{
                    nut_id}_10s_{landcover_short_suffix}.tif''')
            prod_total_potential_task = task_graph.add_task(
                func=mult_rasters,
                args=(
                    ag_task_path_tuple[1], tot_prod_path,
                    prod_total_potential_path),
                target_path_list=[prod_total_potential_path],
                dependent_task_list=[tot_prod_task, ag_task_path_tuple[0]],
                task_name=(
                    f'tot_prod_{nut_id}_10s_{landcover_short_suffix}'))
            schedule_aggregate_to_degree(
                task_graph, prod_total_potential_path, numpy.sum,
                prod_total_potential_task)

            poll_dep_prod_task, poll_dep_prod_path = (
                poll_dep_prod_nut_10s_task_path_map[nut_id])

            prod_poll_dep_potential_nut_scenario_path = os.path.join(
                OUTPUT_DIR,
                f'prod_poll_dep_potential_{nut_id}_10s_'
                f'{landcover_short_suffix}.tif')
            prod_poll_dep_potential_nut_scenario_task = task_graph.add_task(
                func=mult_rasters,
                args=(
                    ag_task_path_tuple[1], poll_dep_prod_path,
                    prod_poll_dep_potential_nut_scenario_path),
                target_path_list=[prod_poll_dep_potential_nut_scenario_path],
                dependent_task_list=[
                    poll_dep_prod_task, ag_task_path_tuple[0]],
                task_name=(
                    f'poll_dep_prod_{nut_id}_'
                    f'10s_{landcover_short_suffix}'))
            (prod_poll_dep_potential_nut_scenario_1d_task,
             prod_poll_dep_potential_nut_scenario_1d_path) = (
                schedule_aggregate_to_degree(
                    task_graph, prod_poll_dep_potential_nut_scenario_path,
                    numpy.sum, prod_poll_dep_potential_nut_scenario_task))

            # pollination independent
            prod_poll_indep_nut_scenario_path = os.path.join(
                OUTPUT_DIR,
                f'prod_poll_indep_{nut_id}_10s_'
                f'{landcover_short_suffix}.tif')
            prod_poll_indep_nut_scenario_task = task_graph.add_task(
                func=subtract_2_rasters,
                args=(
                    prod_total_potential_path,
                    prod_poll_dep_potential_nut_scenario_path,
                    prod_poll_indep_nut_scenario_path),
                target_path_list=[prod_poll_indep_nut_scenario_path],
                dependent_task_list=[
                    prod_total_potential_task,
                    prod_poll_dep_potential_nut_scenario_task],
                task_name=(
                    f'prod_poll_indep_{nut_id}_'
                    f'10s_{landcover_short_suffix}'))
            schedule_aggregate_to_degree(
                task_graph, prod_poll_indep_nut_scenario_path,
                numpy.sum, prod_poll_indep_nut_scenario_task)

            # prod_poll_dep_realized_en|va|fo_10s|1d_cur|ssp1|ssp3|ssp5:
            # pollination-dependent annual production of energy (KJ/yr),
            # vitamin A (IU/yr), and folate (mg/yr) that can be met by wild
            # pollinators due to the proximity of sufficient habitat.
            prod_poll_dep_realized_nut_scenario_path = os.path.join(
                OUTPUT_DIR,
                f'prod_poll_dep_realized_{nut_id}_10s_'
                f'{landcover_short_suffix}.tif')
            prod_poll_dep_realized_nut_scenario_task = task_graph.add_task(
                func=mult_rasters,
                args=(
                    prod_poll_dep_potential_nut_scenario_path,
                    pollinator_suff_hab_path,
                    prod_poll_dep_realized_nut_scenario_path),
                target_path_list=[prod_poll_dep_realized_nut_scenario_path],
                dependent_task_list=[
                    poll_suff_task, prod_poll_dep_potential_nut_scenario_task],
                task_name=(
                    f'prod_poll_dep_realized_{nut_id}_'
                    f'10s_{landcover_short_suffix}'))

            (prod_poll_dep_realized_1d_task,
             prod_poll_dep_realized_nut_scenario_1d_path) = (
                schedule_aggregate_to_degree(
                    task_graph, prod_poll_dep_realized_nut_scenario_path,
                    numpy.sum, prod_poll_dep_realized_nut_scenario_task))
            prod_poll_dep_realized_1d_task_path_map[
                (landcover_short_suffix, nut_id)] = (
                    prod_poll_dep_realized_1d_task,
                    prod_poll_dep_realized_nut_scenario_1d_path)
            summary_raster_path_map[
                f'''prod_poll_dep_realized_{
                    nut_id}_1d_{landcover_short_suffix}'''] = (
                        prod_poll_dep_realized_nut_scenario_1d_path)

            # nat_cont_poll_en|va|fo|avg_10s|1d_cur|ssp1|ssp3|ssp5: "nature's
            # contribution to pollination,"" or the realized
            # pollination-dependent production (prod_poll_dep_realized) over
            # potential pollination-dependent production
            # (prod_poll_dep_potential)
            nat_cont_poll_nut_path = os.path.join(
                OUTPUT_DIR, f'''nat_cont_poll_{nut_id}_10s_{
                    landcover_short_suffix}.tif''')
            nat_cont_poll_nut_task = task_graph.add_task(
                func=calculate_raster_ratio,
                args=(
                    prod_poll_dep_realized_nut_scenario_path,
                    prod_poll_dep_potential_nut_scenario_path,
                    nat_cont_poll_nut_path),
                target_path_list=[nat_cont_poll_nut_path],
                dependent_task_list=[
                    prod_poll_dep_realized_nut_scenario_task,
                    prod_poll_dep_potential_nut_scenario_task],
                task_name=f'''nature contribution 10s {
                    os.path.basename(nat_cont_poll_nut_path)}''')
            nat_cont_task_path_map[nut_id] = (
                nat_cont_poll_nut_task, nat_cont_poll_nut_path)
            nat_cont_poll_nut_1d_path = os.path.join(
                OUTPUT_DIR, f'''nat_cont_poll_{nut_id}_1d_{
                    landcover_short_suffix}.tif''')
            nat_cont_poll_nut_1d_task = task_graph.add_task(
                func=calculate_raster_ratio,
                args=(
                    prod_poll_dep_realized_nut_scenario_1d_path,
                    prod_poll_dep_potential_nut_scenario_1d_path,
                    nat_cont_poll_nut_1d_path),
                target_path_list=[nat_cont_poll_nut_1d_path],
                dependent_task_list=[
                    prod_poll_dep_realized_1d_task,
                    prod_poll_dep_potential_nut_scenario_1d_task],
                task_name=f'''nature contribution {
                    os.path.basename(nat_cont_poll_nut_1d_path)}''')
            summary_raster_path_map[
                f'''nat_cont_poll_{
                    nut_id}_1d_{landcover_short_suffix}'''] = (
                        nat_cont_poll_nut_1d_path)
            prod_poll_dep_realized_1d_task_path_map[
                (landcover_short_suffix, nut_id)] = (
                    nat_cont_poll_nut_1d_task, nat_cont_poll_nut_1d_path)

            # calculate prod_poll_dep_unrealized X1 as
            # prod_total - prod_poll_dep_realized
            prod_poll_dep_unrealized_nut_scenario_path = os.path.join(
                OUTPUT_DIR,
                f'prod_poll_dep_unrealized_{nut_id}_10s_'
                f'{landcover_short_suffix}.tif')
            prod_poll_dep_unrealized_nut_scenario_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([
                    (prod_poll_dep_potential_nut_scenario_path, 1),
                    (prod_poll_dep_realized_nut_scenario_path, 1),
                    (_MULT_NODATA, 'raw'),
                    (_MULT_NODATA, 'raw'),
                    (_MULT_NODATA, 'raw')], sub_two_op,
                    prod_poll_dep_unrealized_nut_scenario_path,
                    gdal.GDT_Float32, -1),
                target_path_list=[prod_poll_dep_unrealized_nut_scenario_path],
                dependent_task_list=[
                    prod_poll_dep_realized_nut_scenario_task,
                    prod_poll_dep_potential_nut_scenario_task],
                task_name=f'''prod poll dep unrealized: {
                    os.path.basename(
                        prod_poll_dep_unrealized_nut_scenario_path)}''')
            (prod_poll_dep_unrealized_nut_scenario_1d_task,
             prod_poll_dep_unrealized_nut_scenario_1d_path) = (
                schedule_aggregate_to_degree(
                    task_graph, prod_poll_dep_unrealized_nut_scenario_path,
                    numpy.sum, prod_poll_dep_unrealized_nut_scenario_task))
            prod_poll_dep_unrealized_1d_task_path_map[
                (landcover_short_suffix, nut_id)] = (
                    prod_poll_dep_unrealized_nut_scenario_1d_task,
                    prod_poll_dep_unrealized_nut_scenario_1d_path)

            summary_raster_path_map[
                f'''prod_poll_dep_unrealized_{
                    nut_id}_1d_{landcover_short_suffix}'''] = (
                        prod_poll_dep_unrealized_nut_scenario_1d_path)

            # calculate prod_total_realized as
            #   prod_total_potential - prod_poll_dep_unrealized
            prod_total_realized_nut_scenario_path = os.path.join(
                OUTPUT_DIR,
                f'prod_total_realized_{nut_id}_10s_'
                f'{landcover_short_suffix}.tif')
            prod_total_realized_nut_scenario_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([
                    (prod_total_potential_path, 1),
                    (prod_poll_dep_unrealized_nut_scenario_path, 1),
                    (_MULT_NODATA, 'raw'),
                    (_MULT_NODATA, 'raw'),
                    (_MULT_NODATA, 'raw')], sub_two_op,
                    prod_total_realized_nut_scenario_path,
                    gdal.GDT_Float32, _MULT_NODATA),
                target_path_list=[prod_total_realized_nut_scenario_path],
                dependent_task_list=[
                    prod_poll_dep_unrealized_nut_scenario_task,
                    prod_total_potential_task],
                task_name=f'''prod poll dep unrealized: {
                    os.path.basename(
                        prod_total_realized_nut_scenario_path)}''')
            (prod_total_realized_nut_1d_scenario_task,
             prod_total_realized_nut_1d_scenario_path) = (
                schedule_aggregate_to_degree(
                    task_graph, prod_total_realized_nut_scenario_path,
                    numpy.sum, prod_total_realized_nut_scenario_task))
            prod_total_realized_1d_task_path_map[
                (landcover_short_suffix, nut_id)] = (
                    prod_total_realized_nut_1d_scenario_task,
                    prod_total_realized_nut_1d_scenario_path)
            summary_raster_path_map[
                f'''prod_total_realized_{
                    nut_id}_1d_{landcover_short_suffix}'''] = (
                        prod_total_realized_nut_1d_scenario_path)

            # poll_cont_prod_en|va|fo|10s|cur|ssp1|ssp3|ssp5: pollination's
            # contribution to production, or the realized
            # pollination-dependent production (prod_poll_dep_realized) over
            # total realized production (prod_total_realized)
            poll_cont_prod_nut_path = os.path.join(
                OUTPUT_DIR, f'''poll_cont_prod_{nut_id}_10s_{
                    landcover_short_suffix}.tif''')
            poll_cont_prod_nut_task = task_graph.add_task(
                func=calculate_raster_ratio,
                args=(
                    prod_poll_dep_realized_nut_scenario_path,
                    prod_total_realized_nut_scenario_path,
                    poll_cont_prod_nut_path),
                target_path_list=[poll_cont_prod_nut_path],
                dependent_task_list=[
                    prod_poll_dep_realized_nut_scenario_task,
                    prod_total_realized_nut_scenario_task],
                task_name=f'''poll cont {
                    os.path.basename(poll_cont_prod_nut_path)}''')
            poll_cont_prod_map[nut_id] = (
                poll_cont_prod_nut_task, poll_cont_prod_nut_path)

            poll_cont_prod_nut_1d_path = os.path.join(
                OUTPUT_DIR, f'''poll_cont_prod_{nut_id}_1d_{
                    landcover_short_suffix}.tif''')
            poll_cont_prod_nut_task = task_graph.add_task(
                func=calculate_raster_ratio,
                args=(
                    prod_poll_dep_realized_nut_scenario_1d_path,
                    prod_total_realized_nut_1d_scenario_path,
                    poll_cont_prod_nut_1d_path),
                target_path_list=[poll_cont_prod_nut_1d_path],
                dependent_task_list=[
                    prod_poll_dep_realized_1d_task,
                    prod_total_realized_nut_1d_scenario_task],
                task_name=f'''poll cont {
                    os.path.basename(poll_cont_prod_nut_1d_path)}''')
            poll_cont_1d_prod_map[nut_id] = (
                poll_cont_prod_nut_task, poll_cont_prod_nut_1d_path)

        poll_cont_prod_avg_path = os.path.join(
            OUTPUT_DIR, f'''poll_cont_prod_avg_10s_{
                landcover_short_suffix}.tif''')
        poll_cont_prod_nut_avg_task = task_graph.add_task(
            func=average_rasters,
            args=tuple(
                [x[1] for x in poll_cont_prod_map.values()] +
                [poll_cont_prod_avg_path]),
            target_path_list=[poll_cont_prod_avg_path],
            dependent_task_list=[
                x[0] for x in poll_cont_prod_map.values()],
            task_name=f'''avg nat cont poll {
                os.path.basename(poll_cont_prod_avg_path)}''')

        poll_cont_prod_1d_avg_path = os.path.join(
            OUTPUT_DIR, f'''poll_cont_prod_avg_1d_{
                landcover_short_suffix}.tif''')
        poll_cont_prod_1d_nut_avg_task = task_graph.add_task(
            func=average_rasters,
            args=tuple(
                [x[1] for x in poll_cont_1d_prod_map.values()] +
                [poll_cont_prod_1d_avg_path]),
            target_path_list=[poll_cont_prod_1d_avg_path],
            dependent_task_list=[
                x[0] for x in poll_cont_1d_prod_map.values()],
            task_name=f'''avg nat cont poll 1d {
                os.path.basename(poll_cont_prod_1d_avg_path)}''')
        summary_raster_path_map[f'''poll_cont_prod_avg_1d_{
            landcover_short_suffix}'''] = poll_cont_prod_1d_avg_path


    # 1.3.    NUTRITION PROVIDED BY WILD POLLINATORS
    # 1.3.1.  Overview
    #   Nutrition provided by wild pollinators on each pixel of agricultural
    # land was calculated according to pollination habitat sufficiency and the
    # pollination-dependent nutrient yields.

    # 1.3.2.  Nutrition production by wild pollinators
    #  Pollinator-dependent nutrient yields at 5 arc minutes were applied to
    # agricultural pixels in the GLOBIO land-use map at 10 arc seconds, and
    # multiplied by wild pollination sufficiency (1 if sufficient, 0 if not) to
    # report pollination-derived nutrient yields in each pixel. We call this
    # "pollinator-derived" instead of "pollination-dependent" because
    # "dependence" is the proportion that yields are reduced if not adequately
    # pollinated. Whether that dependent yield is actually produced is
    # determined by whether there is sufficient natural habitat around the
    # agricultural pixel in question to provide wild pollinators and hence
    # adequate pollination to crops.  These pollinator-derived nutrient yields
    # were then multiplied by the area of the pixel to convert yields to
    # pixel-level production for each nutrient.  Maps of pollination-derived
    # nutrient production for each nutrient in each scenario can be found at
    # (permanent link to output), outputs "poll_serv_..." below.

    spatial_population_scenarios_url = (
        'https://storage.cloud.google.com/ecoshard-root/ipbes/'
        'Spatial_population_scenarios_GeoTIFF_'
        'md5_1c4b6d87cb9a167585e1fc49914248fd.zip')

    spatial_population_scenarios_path = os.path.join(
        ECOSHARD_DIR, 'spatial_population_scenarios',
        os.path.basename(spatial_population_scenarios_url))

    spatial_population_scenarios_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            spatial_population_scenarios_url, GOOGLE_BUCKET_KEY_PATH,
            spatial_population_scenarios_path),
        target_path_list=[spatial_population_scenarios_path],
        task_name=f"""fetch {os.path.basename(
            spatial_population_scenarios_path)}""",
        priority=100)

    spatial_scenarios_pop_zip_touch_file_path = os.path.join(
        os.path.dirname(spatial_population_scenarios_path),
        'Spatial_population_scenarios_GeoTIFF.txt')
    unzip_spatial_population_scenarios_task = task_graph.add_task(
        func=unzip_file,
        args=(
            spatial_population_scenarios_path,
            os.path.dirname(spatial_population_scenarios_path),
            spatial_scenarios_pop_zip_touch_file_path),
        target_path_list=[spatial_scenarios_pop_zip_touch_file_path],
        dependent_task_list=[spatial_population_scenarios_fetch_task],
        task_name=f'unzip Spatial_population_scenarios_GeoTIFF',
        priority=100)

    gpw_urls = {
        'gpw_v4_e_a000_014ft_2010_dens': (
            'https://storage.cloud.google.com/ecoshard-root/ipbes/'
            'gpw_v4_e_a000_014ft_2010_dens_30_sec_md5_'
            '8b2871967b71534d56d2df83e413bf33.tif'),
        'gpw_v4_e_a000_014mt_2010_dens': (
            'https://storage.cloud.google.com/ecoshard-root/ipbes/'
            'gpw_v4_e_a000_014mt_2010_dens_30_sec_md5_'
            '8ddf234eabc0025efd5678776e2ae792.tif'),
        'gpw_v4_e_a065plusft_2010_dens': (
            'https://storage.cloud.google.com/ecoshard-root/ipbes/'
            'gpw_v4_e_a065plusft_2010_dens_30_sec_md5_'
            'c0cfbc8685dbf16cbe520aac963cd6b6.tif'),
        'gpw_v4_e_a065plusmt_2010_dens': (
            'https://storage.cloud.google.com/ecoshard-root/ipbes/'
            'gpw_v4_e_a065plusmt_2010_dens_30_sec_md5_'
            '1d36e79aa083ee25de7295a4a7d10a4b.tif'),
        'gpw_v4_e_atotpopft_2010_dens': (
            'https://storage.cloud.google.com/ecoshard-root/ipbes/'
            'gpw_v4_e_atotpopft_2010_dens_30_sec_md5_'
            '5bbb72a050c76264e1b6a3c7530fedce.tif'),
        'gpw_v4_e_atotpopmt_2010_dens': (
            'https://storage.cloud.google.com/ecoshard-root/ipbes/'
            'gpw_v4_e_atotpopmt_2010_dens_30_sec_md5_'
            '31637ca784b8b917222661d4a915ead6.tif'),
    }

    gpw_task_path_id_map = {}
    for gpw_id, gpw_url in gpw_urls.items():
        gpw_dens_path = os.path.join(
            ECOSHARD_DIR, 'gpw_pop_densities', os.path.basename(gpw_url))
        gpw_fetch_task = task_graph.add_task(
            func=google_bucket_fetch_and_validate,
            args=(gpw_url, GOOGLE_BUCKET_KEY_PATH, gpw_dens_path),
            target_path_list=[gpw_dens_path],
            task_name=f"""fetch {os.path.basename(gpw_dens_path)}""",
            priority=100)

        gpw_count_path = os.path.join(
            CHURN_DIR, 'gpw_count', f"""{gpw_id[:-4]}count.tif""")
        gpw_count_task = task_graph.add_task(
            func=calc_pop_count,
            args=(gpw_dens_path, gpw_count_path),
            target_path_list=[gpw_count_path],
            dependent_task_list=[gpw_fetch_task],
            task_name=f"""pop count {os.path.basename(gpw_count_path)}""")
        gpw_task_path_id_map[f'{gpw_id[:-4]}count'] = (
            gpw_count_task, gpw_count_path)

    # we can't schedule a sum until we have the data, so we join here
    gpw_task_path_id_map['gpw_v4_e_atotpopmt_2010_count'][0].join()
    gpw_task_path_id_map['gpw_v4_e_atotpopft_2010_count'][0].join()
    gpw_1d_path_map = {}
    gpw_1d_task_map = {}
    total_cur_pop_10s_path = os.path.join(
        OUTPUT_DIR, 'gpw_v4_e_atot_pop_10s_cur.tif')
    (total_cur_pop_1d_task, total_cur_pop_1d_path) = (
        schedule_sum_and_aggregate(
            task_graph,
            [gpw_task_path_id_map['gpw_v4_e_atotpopmt_2010_count'][1],
             gpw_task_path_id_map['gpw_v4_e_atotpopft_2010_count'][1]],
            numpy.sum, [
                gpw_task_path_id_map['gpw_v4_e_atotpopmt_2010_count'][0],
                gpw_task_path_id_map['gpw_v4_e_atotpopft_2010_count'][0]],
            total_cur_pop_10s_path))
    gpw_1d_path_map['cur'] = total_cur_pop_1d_path
    gpw_1d_task_map['cur'] = total_cur_pop_1d_task

    # calculate 15-65 population gpw count by subtracting total from
    # 0-14 and 65plus
    for gender_id in ['f', 'm']:
        gpw_v4_e_a15_65t_2010_count_path = os.path.join(
            CHURN_DIR, 'gpw_count',
            f'gpw_v4_e_a015_065{gender_id}_2010_count.tif')
        gpw_15_65f_count_task = task_graph.add_task(
            func=subtract_3_rasters,
            args=(
                gpw_task_path_id_map[
                    f'gpw_v4_e_atotpop{gender_id}t_2010_count'][1],
                gpw_task_path_id_map[
                    f'gpw_v4_e_a000_014{gender_id}t_2010_count'][1],
                gpw_task_path_id_map[
                    f'gpw_v4_e_a065plus{gender_id}t_2010_count'][1],
                gpw_v4_e_a15_65t_2010_count_path),
            target_path_list=[gpw_v4_e_a15_65t_2010_count_path],
            dependent_task_list=[
                gpw_task_path_id_map[
                    f'gpw_v4_e_atotpop{gender_id}t_2010_count'][0],
                gpw_task_path_id_map[
                    f'gpw_v4_e_a000_014{gender_id}t_2010_count'][0],
                gpw_task_path_id_map[
                    f'gpw_v4_e_a065plus{gender_id}t_2010_count'][0]],
            task_name=f'calc gpw 15-65 {gender_id}')
        gpw_task_path_id_map[f'gpw_v4_e_a015_065{gender_id}t_2010_count'] = (
            gpw_15_65f_count_task, gpw_v4_e_a15_65t_2010_count_path)

    # we need to warp SSP rasters to match the GPW rasters
    # we need to calculate 15-65 pop by subtracting 0-14 and 65 plus from tot
    # then we can calculate SSP future for 0-14, 15-65, and 65 plus
    # then we can calculate nutritional needs for cur, and ssp scenarios
    ssp_task_pop_map = {}
    for ssp_id in (1, 3, 5):
        spatial_pop_dir = os.path.join(
            os.path.dirname(spatial_population_scenarios_path),
            'Spatial_population_scenarios_GeoTIFF',
            f'SSP{ssp_id}_GeoTIFF', 'total', 'GeoTIFF')
        cur_ssp_path = os.path.join(spatial_pop_dir, f'ssp{ssp_id}_2010.tif')
        fut_ssp_path = os.path.join(spatial_pop_dir, f'ssp{ssp_id}_2050.tif')

        cur_ssp_warp_path = f'{os.path.splitext(cur_ssp_path)[0]}_gpwwarp.tif'
        fut_ssp_warp_path = f'{os.path.splitext(fut_ssp_path)[0]}_gpwwarp.tif'
        warp_task_list = []
        # we need a canonical base gpw raster to warp to, just grab the
        # "first" one off the dict.
        (gpw_base_tot_count_task, gpw_base_tot_count_path) = (
            gpw_task_path_id_map.values().__iter__().__next__())
        for base_path, warp_path in [
                (cur_ssp_path, cur_ssp_warp_path),
                (fut_ssp_path, fut_ssp_warp_path)]:
            warp_task = task_graph.add_task(
                func=warp_to_raster,
                args=(
                    base_path, gpw_base_tot_count_path, 'bilinear',
                    warp_path),
                dependent_task_list=[
                    gpw_base_tot_count_task,
                    unzip_spatial_population_scenarios_task],
                target_path_list=[warp_path],
                task_name=f'warp to raster {os.path.basename(warp_path)}')
            warp_task_list.append(warp_task)

        gpw_path_list = []
        gpw_task_list = []
        for gpw_id in [
                'gpw_v4_e_a000_014ft_2010_count',
                'gpw_v4_e_a000_014mt_2010_count',
                'gpw_v4_e_a065plusft_2010_count',
                'gpw_v4_e_a065plusmt_2010_count',
                'gpw_v4_e_a015_065ft_2010_count',
                'gpw_v4_e_a015_065mt_2010_count',
                ]:

            gpw_task, gpw_tot_count_path = (
                gpw_task_path_id_map[gpw_id])
            ssp_pop_path = os.path.join(
                CHURN_DIR, 'gpw_ssp_rasters', f'ssp{ssp_id}_{gpw_id}.tif')

            ssp_pop_task = task_graph.add_task(
                func=calculate_future_pop,
                args=(
                    cur_ssp_warp_path, fut_ssp_warp_path, gpw_tot_count_path,
                    ssp_pop_path),
                dependent_task_list=warp_task_list + [gpw_task],
                target_path_list=[ssp_pop_path],
                task_name=f'ssp pop {os.path.basename(ssp_pop_path)}')
            ssp_task_pop_map[(ssp_id, gpw_id)] = (ssp_pop_task, ssp_pop_path)
            gpw_path_list.append(ssp_pop_path)
            gpw_task_list.append(ssp_pop_task)

        total_ssp_pop_1d_path = os.path.join(
            OUTPUT_DIR, f'gpw_v4_e_atot_pop_30s_ssp{ssp_id}.tif')
        (total_ssp_pop_1d_task, total_ssp_pop_1d_path) = (
            schedule_sum_and_aggregate(
                task_graph, gpw_path_list, numpy.sum, gpw_task_list,
                total_ssp_pop_1d_path))
        gpw_1d_path_map[f'gpw_v4_e_atot_pop_30s_ssp{ssp_id}'] = (
            total_ssp_pop_1d_path)
        gpw_1d_task_map[f'gpw_v4_e_atot_pop_30s_ssp{ssp_id}'] = (
            total_ssp_pop_1d_task)

    # 2)
    # calculate the total nutritional needs per pixel for cur ssp1..5 scenario
    # tot_req_en|va|fo_10s|1d_cur|ssp1|ssp3|ssp5

    # this table comes from section "1.4.3 Dietary requirements"
    # units are 'va': Vitamin A (mcg RE)
    # 'fo': folate (mcg DFE)
    # 'en': Energy (kcal) but converted to annual needs

    nutrient_requirements_url = (
        'https://storage.cloud.google.com/ecoshard-root/ipbes/'
        'ipbes_dietary_requirements_RNI_EAR_annual_'
        'md5_27470a02d1fc827704a1cb4ccc765d78.csv')

    nutrient_requirements_table_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(nutrient_requirements_url))

    nutrient_requirements_table_fetch_task = task_graph.add_task(
        func=google_bucket_fetch_and_validate,
        args=(
            nutrient_requirements_url, GOOGLE_BUCKET_KEY_PATH,
            nutrient_requirements_table_path),
        target_path_list=[nutrient_requirements_table_path],
        task_name=f'''fetch {os.path.basename(
            nutrient_requirements_table_path)}''')
    nutrient_requirements_table_fetch_task.join()
    nutritional_needs_df = pandas.read_csv(nutrient_requirements_table_path)
    # this dataframe has ages that need to be converted to the pop raster ids
    nutritional_needs_map = {}
    for age_index, pop_raster_id in (
            ('0-14 F', 'gpw_v4_e_a000_014ft_2010_count'),
            ('0-14 M', 'gpw_v4_e_a000_014mt_2010_count'),
            ('15-64 F', 'gpw_v4_e_a015_065ft_2010_count'),
            ('15-64 M', 'gpw_v4_e_a015_065mt_2010_count'),
            ('65+ F', 'gpw_v4_e_a065plusft_2010_count'),
            ('65+ M', 'gpw_v4_e_a065plusmt_2010_count')):
        gender_row = nutritional_needs_df[
            nutritional_needs_df['age_gender'] == age_index]
        nutritional_needs_map[pop_raster_id] = {
            'va': gender_row['va'].values[0],
            'fo': gender_row['fo'].values[0],
            'en': gender_row['en'].values[0]
        }

    prod_poll_dep_1d_task_path_map = {}
    nut_req_task_path_map = {}
    for nut_id in ('en', 'va', 'fo'):
        # calculate 'cur' needs
        pop_task_path_list, nut_need_list = zip(*[
            (gpw_task_path_id_map[gpw_id],
             nutritional_needs_map[gpw_id][nut_id])
            for gpw_id in nutritional_needs_map])
        pop_task_list, pop_path_list = zip(*pop_task_path_list)

        nut_requirements_path = os.path.join(
            OUTPUT_DIR,
            f'nut_req_{nut_id}_10s_cur.tif')
        nut_requirements_task = task_graph.add_task(
            func=calculate_total_requirements,
            args=(
                pop_path_list, nut_need_list, nut_requirements_path),
            target_path_list=[nut_requirements_path],
            dependent_task_list=pop_task_list,
            task_name=f"""tot nut requirements {
                os.path.basename(nut_requirements_path)}""",
            priority=100,)
        tot_nut_deg_task, tot_nut_deg_path = schedule_aggregate_to_degree(
            task_graph, nut_requirements_path,
            numpy.sum, nut_requirements_task)
        nut_req_task_path_map[('cur', nut_id)] = (
            tot_nut_deg_task, tot_nut_deg_path)
        summary_raster_path_map[f'''nut_req_{nut_id}_1d_cur'''] = (
            tot_nut_deg_path)

        # poll_cont_nut_req_en|va|fo_1d_cur|ssp1|ssp3|ssp5: "nature's
        # contribution to nutrition," the contribution of wild pollination to
        # local nutritional adequacy, as a ratio of the realized
        # pollinator-derived production (prod_poll_dep_realized) to total
        # dietary requirements (nut_req) for energy, vitamin A, and folate
        prod_poll_dep_1d_task, prod_poll_dep_1d_path = (
            prod_poll_dep_realized_1d_task_path_map[('cur', nut_id)])
        poll_cont_nut_req_path = os.path.join(
            OUTPUT_DIR, f'poll_cont_nut_req_{nut_id}_1d_cur.tif')
        poll_cont_nut_task = task_graph.add_task(
            func=calculate_raster_ratio,
            args=(
                prod_poll_dep_1d_path, tot_nut_deg_path,
                poll_cont_nut_req_path),
            target_path_list=[poll_cont_nut_req_path],
            dependent_task_list=[tot_nut_deg_task, prod_poll_dep_1d_task],
            task_name=f'''nature cont to nut {
                os.path.basename(poll_cont_nut_req_path)}''')
        prod_poll_dep_1d_task_path_map[('cur', nut_id)] = (
            poll_cont_nut_task, poll_cont_nut_req_path)
        summary_raster_path_map[
            f'''poll_cont_nut_req_{nut_id}_1d_cur'''] = (
                poll_cont_nut_req_path)

        # calculate ssp needs
        for ssp_id in (1, 3, 5):
            pop_task_path_list, nut_need_list = zip(*[
                (ssp_task_pop_map[(ssp_id, gpw_id)],
                 nutritional_needs_map[gpw_id][nut_id])
                for gpw_id in nutritional_needs_map])
            pop_task_list, pop_path_list = zip(*pop_task_path_list)

            nut_req_path = os.path.join(
                OUTPUT_DIR,
                f'nut_req_{nut_id}_10s_ssp{ssp_id}.tif')
            nut_req_task = task_graph.add_task(
                func=calculate_total_requirements,
                args=(
                    pop_path_list, nut_need_list, nut_req_path),
                target_path_list=[nut_req_path],
                dependent_task_list=pop_task_list,
                task_name=f"""tot nut requirements {
                    os.path.basename(nut_req_path)}""",
                priority=100,)
            tot_nut_deg_task, tot_nut_deg_path = schedule_aggregate_to_degree(
                task_graph, nut_req_path,
                numpy.sum, nut_req_task)
            nut_req_task_path_map[(f'ssp{ssp_id}', nut_id)] = (
                tot_nut_deg_task, tot_nut_deg_path)
            summary_raster_path_map[
                f'''nut_req_{nut_id}_1d_ssp{ssp_id}'''] = (
                    tot_nut_deg_path)

            # poll_cont_nut_req_en|va|fo_1d_cur|ssp1|ssp3|ssp5: "nature's
            # contribution to nutrition," the contribution of wild pollination
            # to local nutritional adequacy, as a ratio of the realized
            # pollinator-derived production (prod_poll_dep_realized) to total
            # dietary requirements (nut_req) for energy, vitamin A, and folate
            prod_poll_dep_1d_task, prod_poll_dep_1d_path = (
                prod_poll_dep_realized_1d_task_path_map[
                    (f'ssp{ssp_id}', nut_id)])
            poll_cont_nut_req_path = os.path.join(
                OUTPUT_DIR, f'poll_cont_nut_req_{nut_id}_1d_ssp{ssp_id}.tif')
            poll_cont_nut_task = task_graph.add_task(
                func=calculate_raster_ratio,
                args=(
                    prod_poll_dep_1d_path, tot_nut_deg_path,
                    poll_cont_nut_req_path),
                target_path_list=[poll_cont_nut_req_path],
                dependent_task_list=[tot_nut_deg_task, prod_poll_dep_1d_task],
                task_name=f'''nature cont to nut {
                    os.path.basename(poll_cont_nut_req_path)}''')
            prod_poll_dep_1d_task_path_map[(f'ssp{ssp_id}', nut_id)] = (
                poll_cont_nut_task, poll_cont_nut_req_path)
            summary_raster_path_map[
                f'''poll_cont_nut_req_{nut_id}_1d_ssp{ssp_id}'''] = (
                    poll_cont_nut_req_path)

    for scenario_id in ('cur', 'ssp1', 'ssp3', 'ssp5'):
        # poll_cont_nut_req_avg_1d_cur|ssp1|ssp3|ssp5: average contribution of
        # wild pollination to local nutritional adequacy, across all three
        # nutrients, with each nutrient capped at 1
        poll_cont_nut_req_avg_1d_path = os.path.join(
            OUTPUT_DIR, f'poll_cont_nut_req_avg_1d_{scenario_id}.tif')
        average_raster_task = task_graph.add_task(
            func=average_rasters,
            args=tuple(
                [prod_poll_dep_1d_task_path_map[(scenario_id, nut_id)][1]
                 for nut_id in ('en', 'fo', 'va')] +
                [poll_cont_nut_req_avg_1d_path]),
            kwargs={'clamp': 1.0},
            target_path_list=[poll_cont_nut_req_avg_1d_path],
            dependent_task_list=[
                prod_poll_dep_1d_task_path_map[(scenario_id, nut_id)][0]
                for nut_id in ('en', 'fo', 'va')],
            task_name=f'''poll cont nut avg 1d {
                os.path.basename(poll_cont_nut_req_avg_1d_path)}''')
        summary_raster_path_map[
            f'''poll_cont_nut_req_avg_1d_{nut_id}_1d_{scenario_id}'''] = (
                poll_cont_nut_req_avg_1d_path)

    # The following came from SQL queries/constructions that becky used to do
    # by hand.
    # record task/path map indexed by landcover_short_suffix
    un_task_path_map = {}
    prod_poll_indep_task_path_map = {}
    for _, (_, landcover_short_suffix) in landcover_data.items():
        poll_dep_task_path_list = []
        nat_cont_poll_task_path_list = []
        prod_poll_dep_unrealized_task_path_list = []
        for nutrient_id in ['en', 'fo', 'va']:
            poll_dep_pot_nut_path = os.path.join(
                OUTPUT_DIR, f'''poll_dep_pot_{
                    nutrient_id}_{landcover_short_suffix}_1d.tif''')

            (prod_poll_dep_realized_task, prod_poll_dep_realized_path) = (
                prod_poll_dep_realized_1d_task_path_map[
                    (landcover_short_suffix, nutrient_id)])
            (prod_poll_dep_unrealized_task, prod_poll_dep_unrealized_path) = (
                prod_poll_dep_unrealized_1d_task_path_map[
                    (landcover_short_suffix, nutrient_id)])
            (prod_total_realized_task, prod_total_realized_path) = (
                prod_total_realized_1d_task_path_map[
                    (landcover_short_suffix, nutrient_id)])

            pol_dep_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=(
                    ((prod_poll_dep_realized_path, 1),
                     (prod_poll_dep_unrealized_path, 1),
                     (prod_total_realized_path, 1),
                     (prod_poll_dep_unrealized_path, 1),
                     (_MULT_NODATA, 'raw')),
                    sum_num_sum_denom,
                    poll_dep_pot_nut_path,
                    _MULT_NODATA, gdal.GDT_Float32),
                target_path_list=[poll_dep_pot_nut_path],
                dependent_task_list=[
                    prod_poll_dep_realized_task,
                    prod_poll_dep_unrealized_task,
                    prod_total_realized_task],
                task_name=f'''calculate poll_dep_pot_{
                    nut_id}_{landcover_short_suffix}''')
            poll_dep_task_path_list.append(
                (pol_dep_task, poll_dep_pot_nut_path))

            (nat_cont_poll_task, nat_cont_poll_path) = (
                prod_poll_dep_realized_1d_task_path_map[
                    (landcover_short_suffix, nutrient_id)])
            nat_cont_poll_task_path_list.append(
                (nat_cont_poll_task, nat_cont_poll_path))
            prod_poll_dep_unrealized_task_path_list.append(
                (prod_poll_dep_unrealized_task,
                    prod_poll_dep_unrealized_path))

            # Relevant population is population in grid cells where
            # pollination-independent production does not already meet or
            # exceed demand
            prod_poll_indep_path = os.path.join(
                OUTPUT_DIR, f'''poll_prod_indep_{
                    nutrient_id}_{landcover_short_suffix}.tif''')
            prod_poll_indep_task = task_graph.add_task(
                func=subtract_2_rasters,
                args=(
                    prod_total_realized_1d_task_path_map[
                        (landcover_short_suffix, nut_id)][1],
                    prod_poll_dep_realized_1d_task_path_map[
                        (landcover_short_suffix, nut_id)][1],
                    prod_poll_indep_path),
                target_path_list=[prod_poll_indep_path],
                dependent_task_list=[
                    prod_total_realized_1d_task_path_map[
                        (landcover_short_suffix, nut_id)][0],
                    prod_poll_dep_realized_1d_task_path_map[
                        (landcover_short_suffix, nut_id)][0]],
                task_name=f'''calculate poll_prod_indep_{
                    nutrient_id}_{landcover_short_suffix}''')
            prod_poll_indep_task_path_map[
                (landcover_short_suffix, nutrient_id)] = (
                    prod_poll_indep_task, prod_poll_indep_path)

        need_path = os.path.join(
            OUTPUT_DIR, f'''need_{landcover_short_suffix}.tif''')
        task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(x[1], 1) for x in poll_dep_task_path_list] +
                [(_MULT_NODATA, 'raw')],
                avg_3_op, need_path, gdal.GDT_Float32, _MULT_NODATA),
            target_path_list=[need_path],
            dependent_task_list=[x[0] for x in poll_dep_task_path_list],
            task_name=f'need_{landcover_short_suffix}')

        ncp_path = os.path.join(
            OUTPUT_DIR, f'''NCP_{landcover_short_suffix}.tif''')
        task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(x[1], 1) for x in nat_cont_poll_task_path_list] +
                [(_MULT_NODATA, 'raw')],
                avg_3_op, ncp_path, gdal.GDT_Float32, _MULT_NODATA),
            target_path_list=[ncp_path],
            dependent_task_list=[x[0] for x in nat_cont_poll_task_path_list],
            task_name=f'create ncp_{landcover_short_suffix}')

        # "Unmet need" is unrealized production, prod_poll_dep_unrealized_
        # averaged across all three nutrients.  In order to average it you
        # first need to normalize: Divide unrealized_va by 486980,
        # unrealized_fo by 132654, and unrealized_en by 3319921; then average.
        # These numbers are the average annual nutrient requirements (in the
        # correct units, that correspond to the production units).
        un_path = os.path.join(
            OUTPUT_DIR, f'UN_{landcover_short_suffix}.tif')
        un_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(x[1], 1) for x in prod_poll_dep_unrealized_task_path_list] +
                [(486980, 'raw'), (132654, 'raw'), (3319921, 'raw')] +
                [(_MULT_NODATA, 'raw')],
                weighted_avg_3_op, un_path, gdal.GDT_Float32, _MULT_NODATA),
            target_path_list=[un_path],
            dependent_task_list=[
                x[0] for x in prod_poll_dep_unrealized_task_path_list],
            task_name=f'ncp_{landcover_short_suffix}')
        un_task_path_map[landcover_short_suffix] = (un_task, un_path)

    for _, (_, landcover_short_suffix) in landcover_data.items():
        cun_path = os.path.join(
            OUTPUT_DIR, f'cUN_{landcover_short_suffix}.tif')

        cun_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                ((un_task_path_map[landcover_short_suffix][1], 1),
                 (un_task_path_map['cur'][1], 1),
                 (_MULT_NODATA, 'raw')),
                prop_diff_op,
                cun_path, gdal.GDT_Float32, _MULT_NODATA),
            target_path_list=[cun_path],
            dependent_task_list=[
                un_task_path_map['cur'][0],
                un_task_path_map[landcover_short_suffix][0]],
            task_name=f'create cUN_{landcover_short_suffix}')

    for scenario_id in ('cur', 'ssp1', 'ssp3', 'ssp5'):
        # relevant_population is base pop where there's needs? I'm inferring
        # this from the SQL queries in the design doc.
        relevant_pop_path = os.path.join(
            OUTPUT_DIR, f'relevant_pop_{landcover_short_suffix}.tif')
        if scenario_id == 'cur':
            gpw_scenario_id = 'cur'
        else:
            gpw_scenario_id = f'gpw_v4_e_atot_pop_30s_{scenario_id}'
        task_graph.add_task(
            func=calc_relevant_pop,
            args=(
                gpw_1d_path_map[gpw_scenario_id],
                prod_poll_indep_task_path_map[(scenario_id, 'en')][1],
                nut_req_task_path_map[(scenario_id, 'en')][1],
                prod_poll_indep_task_path_map[(scenario_id, 'fo')][1],
                nut_req_task_path_map[(scenario_id, 'fo')][1],
                prod_poll_indep_task_path_map[(scenario_id, 'va')][1],
                nut_req_task_path_map[(scenario_id, 'va')][1],
                numpy.bitwise_and, relevant_pop_path
                ),
            target_path_list=[relevant_pop_path],
            dependent_task_list=[
                gpw_1d_task_map[gpw_scenario_id],
                prod_poll_indep_task_path_map[(scenario_id, 'en')][0],
                nut_req_task_path_map[(scenario_id, 'en')][0],
                prod_poll_indep_task_path_map[(scenario_id, 'fo')][0],
                nut_req_task_path_map[(scenario_id, 'fo')][0],
                prod_poll_indep_task_path_map[(scenario_id, 'va')][0],
                nut_req_task_path_map[(scenario_id, 'va')][0]],
            task_name=f'calc_relevant_pop {scenario_id}')

        relevant_min_pop_path = os.path.join(
            OUTPUT_DIR, f'relevant_min_pop_{landcover_short_suffix}.tif')
        task_graph.add_task(
            func=calc_relevant_pop,
            args=(
                gpw_1d_path_map[gpw_scenario_id],
                prod_poll_indep_task_path_map[(scenario_id, 'en')][1],
                nut_req_task_path_map[(scenario_id, 'en')][1],
                prod_poll_indep_task_path_map[(scenario_id, 'fo')][1],
                nut_req_task_path_map[(scenario_id, 'fo')][1],
                prod_poll_indep_task_path_map[(scenario_id, 'va')][1],
                nut_req_task_path_map[(scenario_id, 'va')][1],
                numpy.bitwise_or, relevant_min_pop_path
                ),
            target_path_list=[relevant_min_pop_path],
            dependent_task_list=[
                gpw_1d_task_map[gpw_scenario_id],
                prod_poll_indep_task_path_map[(scenario_id, 'en')][0],
                nut_req_task_path_map[(scenario_id, 'en')][0],
                prod_poll_indep_task_path_map[(scenario_id, 'fo')][0],
                nut_req_task_path_map[(scenario_id, 'fo')][0],
                prod_poll_indep_task_path_map[(scenario_id, 'va')][0],
                nut_req_task_path_map[(scenario_id, 'va')][0]],
            task_name=f'calc_relevant_pop {scenario_id}')

    task_graph.close()
    task_graph.join()
    LOGGER.error("don't forget to not return here okay?")
    return
    countries_myregions_df = pandas.read_csv(
        'countries_myregions_final_md5_7e35a0775335f9aaf9a28adbac0b8895.csv',
        usecols=['country', 'myregions'], sep=',')
    country_to_region_dict = {
        row[1][1]: row[1][0] for row in countries_myregions_df.iterrows()}

    grid_shapefile_path = os.path.join(ECOSHARD_DIR, 'grid_1_degree.shp')
    grid_shapefile_vector = gdal.OpenEx(grid_shapefile_path, gdal.OF_VECTOR)
    geopackage_driver = gdal.GetDriverByName('GPKG')
    target_summary_shapefile_path = os.path.join(
        OUTPUT_DIR, f'ipbes_pollination_summary_{HG_SUFFIX}.gpkg')
    target_summary_grid_vector = geopackage_driver.CreateCopy(
        target_summary_shapefile_path, grid_shapefile_vector)
    target_summary_grid_layer = target_summary_grid_vector.GetLayer()

    tm_world_borders_path = os.path.join(
        ECOSHARD_DIR, 'TM_WORLD_BORDERS-0.3.shp')
    LOGGER.debug("build country spatial index")
    country_rtree, country_geom_list = build_spatial_index(
        tm_world_borders_path)
    LOGGER.debug("build hunger spatial index")
    hunger_path = os.path.join(ECOSHARD_DIR, 'hunger.shp')
    hunger_rtree, hunger_geom_list = build_spatial_index(hunger_path)

    country_vector = gdal.OpenEx(tm_world_borders_path, gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()

    hunger_vector = gdal.OpenEx(hunger_path, gdal.OF_VECTOR)
    hunger_layer = hunger_vector.GetLayer()

    for field_name in ['country', 'region']:
        target_summary_grid_layer.CreateField(
            ogr.FieldDefn(field_name, ogr.OFTString))
    for field_name in itertools.chain(
            ['PCTU5', 'UW'], summary_raster_path_map, gpw_1d_path_map):
        target_summary_grid_layer.CreateField(
            ogr.FieldDefn(field_name, ogr.OFTReal))

    # this one does the rasters, we can load them all at once because they
    # are so small, then we can do each feature individually
    array_feature_id_map = {}
    for field_name, raster_path in itertools.chain(
            summary_raster_path_map.items(),
            gpw_1d_path_map.items()):
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        x_size = band.XSize
        y_size = band.YSize
        array_feature_id_map[field_name] = (
            band.ReadAsArray(), band.GetNoDataValue(),
            raster.GetGeoTransform())
        band = None
        raster = None

    # process one grid at a time
    last_time = time.time()
    total_grid_count = target_summary_grid_layer.GetFeatureCount()
    for grid_index, grid_feature in enumerate(target_summary_grid_layer):

        current_time = time.time()
        if current_time - last_time > 5.0:
            last_time = current_time
            LOGGER.debug(
                "processing grid intersection %.2f%% complete",
                (100.0 * (grid_index+1)) / total_grid_count)

        grid_feature_geom_ref = grid_feature.GetGeometryRef()
        centroid = grid_feature_geom_ref.Centroid()

        grid_feature_geom = shapely.wkb.loads(
            grid_feature.GetGeometryRef().ExportToWkb())

        for country_index in country_rtree.intersection(
                grid_feature_geom.bounds):
            if country_geom_list[country_index].intersects(grid_feature_geom):
                country_name = country_layer.GetFeature(
                    country_index).GetField('name')
                grid_feature.SetField('country', country_name)
                try:
                    grid_feature.SetField(
                        'region', country_to_region_dict[country_name])
                except KeyError:
                    grid_feature.SetField('region', 'UNKNOWN')
                break

        for hunger_index in hunger_rtree.intersection(
                grid_feature_geom.bounds):
            if hunger_geom_list[hunger_index].intersects(grid_feature_geom):
                hunger_feature = hunger_layer.GetFeature(hunger_index)
                grid_feature.SetField(
                    'PCTU5', hunger_feature.GetField('PCTU5'))
                grid_feature.SetField('UW', hunger_feature.GetField('UW'))

        long_coord = centroid.GetX()
        lat_coord = centroid.GetY()

        for field_name, (raster_array, raster_nodata, gt) in (
                array_feature_id_map.items()):
            x_coord = int((long_coord - gt[0]) / gt[1])
            if not 0 <= x_coord < x_size:
                continue
            y_coord = int((lat_coord - gt[3]) / gt[5])
            if not 0 <= y_coord < y_size:
                continue
            pixel_value = raster_array[y_coord, x_coord]
            if raster_nodata is not None and ~numpy.isclose(
                    pixel_value, raster_nodata):
                grid_feature.SetField(field_name, float(pixel_value))
        target_summary_grid_layer.SetFeature(grid_feature)

    # prod_poll_dep_realized_en|va|fo_1d_cur|ssp1|ssp3|ssp5
    # prod_poll_dep_unrealized_en|va|fo_1d_cur|ssp1|ssp3|ssp5
    # prod_total_realized_en|va|fo_1d_cur|ssp1|ssp3|ssp5
    # nut_req_en|va|fo_1d_cur|ssp1|ssp3|ssp5
    # nat_cont_poll_avg_1d_cur|ssp1|ssp3|ssp5
    # poll_cont_prod_avg_1d_cur|ssp1|ssp3|ssp5
    # poll_cont_nut_req_avg_1d_cur|ssp1|ssp3|ssp5cur|ssp1|ssp3|ssp5gpwpop

    # END MAIN


def build_spatial_index(vector_path):
    """Build an rtree/geom list tuple from ``vector_path``."""
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    geom_index = rtree.index.Index()
    geom_list = []
    for index in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(index)
        geom = feature.GetGeometryRef()
        shapely_geom = shapely.wkb.loads(geom.ExportToWkb())
        shapely_prep_geom = shapely.prepared.prep(shapely_geom)
        geom_list.append(shapely_prep_geom)
        geom_index.insert(index, shapely_geom.bounds)

    return geom_index, geom_list


def calculate_total_requirements(
        pop_path_list, nut_need_list, target_path):
    """Calculate total nutrient requirements.

    Create a new raster by summing all rasters in `pop_path_list` multiplied
    by their corresponding scalar in `nut_need_list`.

    Parameters:
        pop_path_list (list of str): list of paths to population counts.
        nut_need_list (list): list of scalars that correspond in order to
            the per-count nutrient needs of `pop_path_list`.
        target_path (str): path to target file.

    Return:
        None.

    """
    nodata = -1
    pop_nodata = pygeoprocessing.get_raster_info(
        pop_path_list[0])['nodata'][0]

    def mult_and_sum(*arg_list):
        """Arg list is an (array0, scalar0, array1, scalar1,...) list.

        Returns:
            array0*scalar0 + array1*scalar1 + .... but ignore nodata.

        """
        result = numpy.empty(arg_list[0].shape, dtype=numpy.float32)
        result[:] = nodata
        array_stack = numpy.array(arg_list[0::2])
        scalar_list = numpy.array(arg_list[1::2])
        # make a valid mask as big as a single array
        valid_mask = numpy.logical_and.reduce(
            array_stack != pop_nodata, axis=0)

        # mask out all invalid elements but reshape so there's still the same
        # number of arrays
        valid_array_elements = (
            array_stack[numpy.broadcast_to(valid_mask, array_stack.shape)])
        array_stack = None

        # sometimes this array is empty, check first before reshaping
        if valid_array_elements.size != 0:
            valid_array_elements = valid_array_elements.reshape(
                -1, numpy.count_nonzero(valid_mask))
            # multiply each element of the scalar with each row of the valid
            # array stack, then sum along the 0 axis to get the result
            result[valid_mask] = numpy.sum(
                (valid_array_elements.T * scalar_list).T, axis=0)
        scalar_list = None
        valid_mask = None
        valid_array_elements = None
        return result

    pygeoprocessing.raster_calculator(list(itertools.chain(*[
        ((path, 1), (scalar, 'raw')) for path, scalar in zip(
            pop_path_list, nut_need_list)])), mult_and_sum, target_path,
        gdal.GDT_Float32, nodata)


def sub_two_op(a_array, b_array, a_nodata, b_nodata, target_nodata):
    """Subtract a from b and ignore nodata."""
    result = numpy.empty_like(a_array)
    result[:] = target_nodata
    valid_mask = (a_array != a_nodata) & (b_array != b_nodata)
    result[valid_mask] = a_array[valid_mask] - b_array[valid_mask]
    return result


def average_rasters(*raster_list, clamp=None):
    """Average rasters in raster list except write to the last one.

    Parameters:
        raster_list (list of string): list of rasters to average over.
        clamp (float): value to clamp the individual raster to before the
            average.

    Returns:
        None.

    """
    nodata_list = [
        pygeoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_list[:-1]]
    target_nodata = -1.

    def average_op(*array_list):
        result = numpy.empty_like(array_list[0])
        result[:] = target_nodata
        valid_mask = numpy.ones(result.shape, dtype=numpy.bool)
        clamped_list = []
        for array, nodata in zip(array_list, nodata_list):
            valid_mask &= array != nodata
            if clamp:
                clamped_list.append(
                    numpy.where(array > clamp, clamp, array))
            else:
                clamped_list.append(array)

        if valid_mask.any():
            array_stack = numpy.stack(clamped_list)
            result[valid_mask] = numpy.average(
                array_stack[numpy.broadcast_to(
                    valid_mask, array_stack.shape)].reshape(
                        len(array_list), -1), axis=0)
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list[:-1]], average_op,
        raster_list[-1], gdal.GDT_Float32, target_nodata)


def subtract_2_rasters(
        raster_path_a, raster_path_b, target_path):
    """Calculate target = a-b and ignore nodata."""
    a_nodata = pygeoprocessing.get_raster_info(raster_path_a)['nodata'][0]
    b_nodata = pygeoprocessing.get_raster_info(raster_path_b)['nodata'][0]
    target_nodata = -9999

    def sub_op(a_array, b_array):
        """Sub a-b-c as arrays."""
        result = numpy.empty(a_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (
            (a_array != a_nodata) &
            (b_array != b_nodata))
        result[valid_mask] = (
            a_array[valid_mask] - b_array[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(raster_path_a, 1), (raster_path_b, 1)],
        sub_op, target_path, gdal.GDT_Float32, target_nodata)


def subtract_3_rasters(
        raster_path_a, raster_path_b, raster_path_c, target_path):
    """Calculate target = a-b-c and ignore nodata."""
    a_nodata = pygeoprocessing.get_raster_info(raster_path_a)['nodata'][0]
    b_nodata = pygeoprocessing.get_raster_info(raster_path_b)['nodata'][0]
    c_nodata = pygeoprocessing.get_raster_info(raster_path_c)['nodata'][0]
    target_nodata = -9999

    def sub_op(a_array, b_array, c_array):
        """Sub a-b-c as arrays."""
        result = numpy.empty(a_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (
            (a_array != a_nodata) &
            (b_array != b_nodata) &
            (c_array != c_nodata))
        result[valid_mask] = (
            a_array[valid_mask] - b_array[valid_mask] - c_array[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(raster_path_a, 1), (raster_path_b, 1), (raster_path_c, 1)],
        sub_op, target_path, gdal.GDT_Float32, target_nodata)


def calculate_raster_ratio(
        raster_a_path, raster_b_path, target_raster_path, clamp_to=None):
    """Calculate the ratio of a:b and ignore nodata and divide by 0.

    Parameters:
        raster_a_path (string): path to numerator of ratio
        raster_b_path (string): path to denominator of ratio
        target_raster_path (string): path to desired target raster that will
            use a nodata value of -1.
        clamp_to (numeric): if not None, the ratio is capped to this value.

    Returns:
        None.

    """
    temp_dir = tempfile.mkdtemp()
    raster_a_aligned_path = os.path.join(
        temp_dir, os.path.basename(raster_a_path))
    raster_b_aligned_path = os.path.join(
        temp_dir, os.path.basename(raster_b_path))
    target_pixel_size = pygeoprocessing.get_raster_info(
        raster_a_path)['pixel_size']
    pygeoprocessing.align_and_resize_raster_stack(
        [raster_a_path, raster_b_path],
        [raster_a_aligned_path, raster_b_aligned_path], ['near']*2,
        target_pixel_size, 'intersection')

    nodata_a = pygeoprocessing.get_raster_info(
        raster_a_aligned_path)['nodata'][0]
    nodata_b = pygeoprocessing.get_raster_info(
        raster_b_aligned_path)['nodata'][0]
    target_nodata = -1.

    def ratio_op(array_a, array_b):
        result = numpy.empty(array_a.shape, dtype=numpy.float32)
        result[:] = target_nodata
        zero_mask = numpy.isclose(array_b, 0.)
        valid_mask = (
            ~numpy.isclose(array_a, nodata_a) &
            ~numpy.isclose(array_b, nodata_b) &
            ~zero_mask)
        result[valid_mask] = array_a[valid_mask] / array_b[valid_mask]
        if clamp_to:
            result[valid_mask & (result > clamp_to)] = clamp_to
        result[zero_mask] = 0.0
        return result

    pygeoprocessing.raster_calculator(
        [(raster_a_aligned_path, 1), (raster_b_aligned_path, 1)], ratio_op,
        target_raster_path, gdal.GDT_Float32, target_nodata)

    shutil.rmtree(temp_dir, ignore_errors=True)


def warp_to_raster(
        base_raster_path, canonical_raster_path, resample_method,
        target_raster_path):
    """Warp base raster to canonical example.

    Parameters:
        base_raster_path (str): path to base raster
        canonical_raster_path (str),
        resample_method (str): one of nearest, bilinear, cubic, cubic_spline,
            lanczos, average, mode, max, min, med, q1, q3.
        target_raster_path (str): path to target warped raster.

    Returns:
        None.

    """
    canonical_raster_info = pygeoprocessing.get_raster_info(
        canonical_raster_path)
    pygeoprocessing.warp_raster(
        base_raster_path, canonical_raster_info['pixel_size'],
        target_raster_path,
        resample_method, target_bb=canonical_raster_info['bounding_box'],
        n_threads=2)


def calculate_future_pop(
        cur_ssp_path, fut_ssp_path, gpw_tot_count_path, target_ssp_pop_path):
    """Calculate future population raster.

    Multiply `gpw_tot_count` by the fut_ssp/cur_ssp ratio.
    """
    target_nodata = -1
    ssp_cur_nodata = pygeoprocessing.get_raster_info(
        cur_ssp_path)['nodata'][0]
    ssp_fut_nodata = pygeoprocessing.get_raster_info(
        fut_ssp_path)['nodata'][0]
    count_nodata = pygeoprocessing.get_raster_info(
        gpw_tot_count_path)['nodata'][0]

    def _future_pop_op(cur_array, fut_array, count_array):
        """Calculate future pop by dividing fut/cur*cur_count."""
        result = numpy.empty(cur_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        zero_mask = cur_array == 0
        valid_mask = (
            (cur_array != ssp_cur_nodata) &
            (fut_array != ssp_fut_nodata) &
            (count_array != count_nodata) & ~zero_mask)
        result[valid_mask] = (
            (fut_array[valid_mask] / cur_array[valid_mask]).astype(
                numpy.float32) * count_array[valid_mask])
        # assume if denominator is 0 we don't want to mask anything out, just
        # get it working
        result[zero_mask] = 1.0
        return result

    pygeoprocessing.raster_calculator(
        [(cur_ssp_path, 1), (fut_ssp_path, 1), (gpw_tot_count_path, 1)],
        _future_pop_op, target_ssp_pop_path, gdal.GDT_Float32, target_nodata)


def calc_pop_count(gpw_dens_path, gpw_count_path):
    """Calculate population count from density.

    Parameters:
        gpw_dens_path (string): path to density raster in units of
            people / km^2.
        gpw_count_path (string): path to target raster to generate number of
            people / pixel.

    """
    gpw_dens_info = pygeoprocessing.get_raster_info(gpw_dens_path)
    y_lat_array = numpy.linspace(
        gpw_dens_info['geotransform'][3],
        gpw_dens_info['geotransform'][3] +
        gpw_dens_info['geotransform'][5] *
        gpw_dens_info['raster_size'][1],
        gpw_dens_info['raster_size'][1])

    # `area_of_pixel` is in m^2, convert to km^2
    y_km2_array = area_of_pixel(
        abs(gpw_dens_info['geotransform'][1]),
        y_lat_array) * 1e-6
    y_km2_column = y_km2_array.reshape((y_km2_array.size, 1))

    nodata = gpw_dens_info['nodata'][0]
    try:
        os.makedirs(os.path.dirname(gpw_count_path))
    except OSError:
        pass
    pygeoprocessing.raster_calculator(
        [(gpw_dens_path, 1), y_km2_column, (nodata, 'raw')],
        density_to_value_op, gpw_count_path, gdal.GDT_Float32,
        nodata)


def create_radial_convolution_mask(
        pixel_size_degree, radius_meters, kernel_filepath):
    """Create a radial mask to sample pixels in convolution filter.

    Parameters:
        pixel_size_degree (float): size of pixel in degrees.
        radius_meters (float): desired size of radial mask in meters.

    Returns:
        A 2D numpy array that can be used in a convolution to aggregate a
        raster while accounting for partial coverage of the circle on the
        edges of the pixel.

    """
    degree_len_0 = 110574  # length at 0 degrees
    degree_len_60 = 111412  # length at 60 degrees
    pixel_size_m = pixel_size_degree * (degree_len_0 + degree_len_60) / 2.0
    pixel_radius = numpy.ceil(radius_meters / pixel_size_m)
    n_pixels = (int(pixel_radius) * 2 + 1)
    sample_pixels = 200
    mask = numpy.ones((sample_pixels * n_pixels, sample_pixels * n_pixels))
    mask[mask.shape[0]//2, mask.shape[0]//2] = 0
    distance_transform = scipy.ndimage.morphology.distance_transform_edt(mask)
    mask = None
    stratified_distance = distance_transform * pixel_size_m / sample_pixels
    distance_transform = None
    in_circle = numpy.where(stratified_distance <= 2000.0, 1.0, 0.0)
    stratified_distance = None
    reshaped = in_circle.reshape(
        in_circle.shape[0] // sample_pixels, sample_pixels,
        in_circle.shape[1] // sample_pixels, sample_pixels)
    kernel_array = numpy.sum(reshaped, axis=(1, 3)) / sample_pixels**2
    normalized_kernel_array = kernel_array / numpy.sum(kernel_array)
    reshaped = None

    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath.encode('utf-8'), n_pixels, n_pixels, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([-180, 1, 0, 90, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    kernel_raster.SetProjection(srs.ExportToWkt())
    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(NODATA)
    kernel_band.WriteArray(normalized_kernel_array)


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


def threshold_select_raster(
        base_raster_path, select_raster_path, threshold_val, target_path):
    """Select `select` if `base` >= `threshold_val`.

    Parameters:
        base_raster_path (string): path to single band raster that will be
            used to determine the threshold mask to select from
            `select_raster_path`.
        select_raster_path (string): path to single band raster to pass
            through to target if aligned `base` pixel is >= `threshold_val`
            0 otherwise, or nodata if base == nodata. Must be the same
            shape as `base_raster_path`.
        threshold_val (numeric): value to use as threshold cutoff
        target_path (string): path to desired output raster, raster is a
            byte type with same dimensions and projection as
            `base_raster_path`. A pixel in this raster will be `select` if
            the corresponding pixel in `base_raster_path` is >=
            `threshold_val`, 0 otherwise or nodata if `base` == nodata.

    Returns:
        None.

    """
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    target_nodata = -9999.

    def threshold_select_op(
            base_array, select_array, threshold_val, base_nodata,
            target_nodata):
        result = numpy.empty(select_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (base_array != base_nodata) & (select_array == 1)
        result[valid_mask] = numpy.interp(
            base_array[valid_mask], [0, threshold_val], [0.0, 1.0], 0, 1)
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (select_raster_path, 1),
         (threshold_val, 'raw'), (base_nodata, 'raw'),
         (target_nodata, 'raw')], threshold_select_op,
        target_path, gdal.GDT_Float32, target_nodata, gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
            'PREDICTOR=2', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
            'NUM_THREADS=2'))


def mask_raster(base_path, codes, target_path):
    """Mask `base_path` to 1 where values are in codes. 0 otherwise.

    Parameters:
        base_path (string): path to single band integer raster.
        codes (list): list of integer or tuple integer pairs. Membership in
            `codes` or within the inclusive range of a tuple in `codes`
            is sufficient to mask the corresponding raster integer value
            in `base_path` to 1 for `target_path`.
        target_path (string): path to desired mask raster. Any corresponding
            pixels in `base_path` that also match a value or range in
            `codes` will be masked to 1 in `target_path`. All other values
            are 0.

    Returns:
        None.

    """
    code_list = numpy.array([
        item for sublist in [
            range(x[0], x[1]+1) if isinstance(x, tuple) else [x]
            for x in codes] for item in sublist])
    LOGGER.debug(f'expanded code array {code_list}')

    base_nodata = pygeoprocessing.get_raster_info(base_path)['nodata'][0]
    mask_nodata = 2

    def mask_codes_op(base_array, codes_array):
        """Return a bool raster if value in base_array is in codes_array."""
        result = numpy.empty(base_array.shape, dtype=numpy.int8)
        result[:] = mask_nodata
        valid_mask = base_array != base_nodata
        result[valid_mask] = numpy.isin(
            base_array[valid_mask], codes_array)
        return result

    pygeoprocessing.raster_calculator(
        [(base_path, 1), (code_list, 'raw')], mask_codes_op, target_path,
        gdal.GDT_Byte, 2, gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
            'PREDICTOR=2', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
            'NUM_THREADS=2'))


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


def _make_logger_callback(message):
    """Build a timed logger callback that prints `message` replaced.

    Parameters:
        message (string): a string that expects a %f placement variable,
            for % complete.

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, psz_message, p_progress_arg):
        """Log updates using GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                LOGGER.info(message, df_complete * 100)
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0

    return logger_callback


def total_yield_op(
        yield_nodata, pollination_yield_factor_list,
        *crop_yield_harea_array_list):
    """Calculate total yield.

    Parameters:
        yield_nodata (numeric): nodata value for the arrays in
            ``crop_yield_array_list```.
        pollination_yield_factor_list (list of float): list of non-refuse
            proportion of yield that is pollination dependent.
        crop_yield_harea_array_list (list of numpy.ndarray): list of length
            n of 2D arrays of n/2 yield (tons/Ha) for crops that correlate in
            order with the ``pollination_yield_factor_list`` followed by
            n/2 harea (proportional area) of those crops.

    """
    result = numpy.empty(
        crop_yield_harea_array_list[0].shape, dtype=numpy.float32)
    result[:] = 0.0
    all_valid = numpy.zeros(result.shape, dtype=numpy.bool)

    n_crops = len(crop_yield_harea_array_list) // 2
    for crop_index in range(n_crops):
        crop_array = crop_yield_harea_array_list[crop_index]
        harea_array = crop_yield_harea_array_list[crop_index + n_crops]
        valid_mask = crop_array != yield_nodata
        all_valid |= valid_mask
        result[valid_mask] += (
            crop_array[valid_mask] * harea_array[valid_mask] *
            pollination_yield_factor_list[crop_index])
    result[~all_valid] = yield_nodata
    return result


def density_to_value_op(density_array, area_array, density_nodata):
    """Calculate production.

    Parameters:
        density_array (numpy.ndarray): array of densities / area in
            ``area_array``.
        area_array (numpy.ndarray): area of each cell that corresponds with
            ``density_array``.
        density_ndoata (numeric): nodata value of the ``density_array``.

    """
    result = numpy.empty(density_array.shape, dtype=numpy.float32)
    result[:] = density_nodata
    valid_mask = density_array != density_nodata
    result[valid_mask] = density_array[valid_mask] * area_array[valid_mask]
    return result


def create_prod_nutrient_raster(
        crop_nutrient_df_path, nutrient_name, yield_and_harea_raster_dir,
        consider_pollination, sample_target_raster_path,
        target_10km_yield_path, target_10s_yield_path,
        target_10s_production_path):
    """Create total production & yield for a nutrient for all crops.

    Parameters:
        crop_nutrient_df_path (str): path to CSV with at least the
            column `filenm`, `nutrient_name`, `Percent refuse crop`, and
            `Pollination dependence crop`.
        nutrient_name (str): nutrient name to use to index into the crop
            data frame.
        yield_and_harea_raster_dir (str): path to a directory that has files
            of the format `[crop_name]_yield.tif` and
            `[crop_name]_harea.tif` where `crop_name` is a value
            in the `filenm` column of `crop_nutrient_df`.
        consider_pollination (bool): if True, multiply yields by pollinator
            dependence ratio.
        sample_target_raster_path (path): path to a file that has the raster
            pixel size and dimensions of the desired
            `target_10s_production_path`.
        sample_target_fetch_task (Task): must be complete before
            `sample_target_raster_path` is available.
        target_10km_yield_path (str): path to target raster that will
            contain total yield (tons/Ha)
        target_10s_yield_path (str): path to a resampled
            `target_10km_yield_path` at 10s resolution.
        target_10s_production_path (str): path to target raster that will
            contain a per-pixel amount of pollinator produced `nutrient_name`
            calculated as the sum(
                crop_yield_map * (100-Percent refuse crop) *
                (Pollination dependence crop) * nutrient) * (ha / pixel map))

    Returns:
        None.

    """
    for path in [
            target_10km_yield_path, target_10s_yield_path,
            target_10s_production_path]:
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass
    crop_nutrient_df = pandas.read_csv(crop_nutrient_df_path)
    yield_raster_path_list = []
    harea_raster_path_list = []
    pollination_yield_factor_list = []
    for _, row in crop_nutrient_df.iterrows():
        yield_raster_path = os.path.join(
            yield_and_harea_raster_dir, f"{row['filenm']}_yield.tif")
        harea_raster_path = os.path.join(
            yield_and_harea_raster_dir, f"{row['filenm']}_harea.tif")
        if os.path.exists(yield_raster_path):
            yield_raster_path_list.append(yield_raster_path)
            harea_raster_path_list.append(harea_raster_path)
            pollination_yield_factor_list.append(
                (1. - row['Percent refuse'] / 100.) * row[nutrient_name])
            if consider_pollination:
                pollination_yield_factor_list[-1] *= (
                    row['Pollination dependence'])
        else:
            raise ValueError(f"not found {yield_raster_path}")

    sample_target_raster_info = pygeoprocessing.get_raster_info(
        sample_target_raster_path)

    yield_raster_info = pygeoprocessing.get_raster_info(
        yield_raster_path_list[0])
    yield_nodata = yield_raster_info['nodata'][0]

    pygeoprocessing.raster_calculator(
        [(yield_nodata, 'raw'), (pollination_yield_factor_list, 'raw')] +
        [(x, 1) for x in yield_raster_path_list + harea_raster_path_list],
        total_yield_op, target_10km_yield_path, gdal.GDT_Float32,
        yield_nodata),

    y_lat_array = numpy.linspace(
        sample_target_raster_info['geotransform'][3],
        sample_target_raster_info['geotransform'][3] +
        sample_target_raster_info['geotransform'][5] *
        sample_target_raster_info['raster_size'][1],
        sample_target_raster_info['raster_size'][1])

    y_ha_array = area_of_pixel(
        abs(sample_target_raster_info['geotransform'][1]),
        y_lat_array) / 10000.0
    y_ha_column = y_ha_array.reshape((y_ha_array.size, 1))

    pygeoprocessing.warp_raster(
        target_10km_yield_path,
        sample_target_raster_info['pixel_size'], target_10s_yield_path,
        'cubicspline', target_bb=sample_target_raster_info['bounding_box'],
        n_threads=2)

    # multiplying the ha_array by 1e4 because the of yield are in
    # nutrient / 100g and yield is in Mg / ha.
    pygeoprocessing.raster_calculator(
        [(target_10s_yield_path, 1), y_ha_column * 1e4,
         (yield_nodata, 'raw')], density_to_value_op,
        target_10s_production_path, gdal.GDT_Float32, yield_nodata)


def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = numpy.sqrt(1-(b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*numpy.sin(numpy.radians(f))
        zp = 1 + e*numpy.sin(numpy.radians(f))
        area_list.append(
            numpy.pi * b**2 * (
                numpy.log(zp/zm) / (2*e) +
                numpy.sin(numpy.radians(f)) / (zp*zm)))
    return pixel_size / 360. * (area_list[0]-area_list[1])


def _mult_raster_op(array_a, array_b, nodata_a, nodata_b, target_nodata):
    """Multiply a by b and skip nodata."""
    result = numpy.empty(array_a.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = (array_a != nodata_a) & (array_b != nodata_b)
    result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
    return result


def mult_rasters(raster_a_path, raster_b_path, target_path):
    """Multiply a by b and skip nodata."""
    raster_info_a = pygeoprocessing.get_raster_info(raster_a_path)
    raster_info_b = pygeoprocessing.get_raster_info(raster_b_path)

    nodata_a = raster_info_a['nodata'][0]
    nodata_b = raster_info_b['nodata'][0]

    if raster_info_a['raster_size'] != raster_info_b['raster_size']:
        aligned_raster_a_path = (
            target_path + os.path.basename(raster_a_path) + '_aligned.tif')
        aligned_raster_b_path = (
            target_path + os.path.basename(raster_b_path) + '_aligned.tif')
        pygeoprocessing.align_and_resize_raster_stack(
            [raster_a_path, raster_b_path],
            [aligned_raster_a_path, aligned_raster_b_path],
            ['near'] * 2, raster_info_a['pixel_size'], 'intersection')
        raster_a_path = aligned_raster_a_path
        raster_b_path = aligned_raster_b_path

    pygeoprocessing.raster_calculator(
        [(raster_a_path, 1), (raster_b_path, 1), (nodata_a, 'raw'),
         (nodata_b, 'raw'), (_MULT_NODATA, 'raw')], _mult_raster_op,
        target_path, gdal.GDT_Float32, _MULT_NODATA)


def add_op(target_nodata, *array_list):
    """Add & return arrays in ``array_list`` but ignore ``target_nodata``."""
    result = numpy.zeros(array_list[0].shape, dtype=numpy.float32)
    valid_mask = numpy.zeros(result.shape, dtype=numpy.bool)
    for array in array_list:
        # nodata values will be < 0
        local_valid_mask = array >= 0
        valid_mask |= local_valid_mask
        result[local_valid_mask] += array[local_valid_mask]
    result[~valid_mask] = target_nodata
    return result


def schedule_sum_and_aggregate(
        task_graph, base_raster_path_list, aggregate_func,
        base_raster_task_list, target_10s_path):
    """Sum all rasters in `base_raster_path` and aggregate to degree."""
    path_list = [(path, 1) for path in base_raster_path_list]

    # this is the only way to get around that we need to check raster size
    for task in base_raster_task_list:
        task.join()

    raster_size_set = set(
        [pygeoprocessing.get_raster_info(path)['raster_size']
         for path in base_raster_path_list])

    dependent_task_list = list(base_raster_task_list)
    if len(raster_size_set) > 1:
        aligned_path_list = [
            target_10s_path + os.path.basename(path) + '_aligned.tif'
            for path in base_raster_path_list]
        align_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            args=(
                base_raster_path_list,
                aligned_path_list,
                ['near'] * len(base_raster_path_list),
                pygeoprocessing.get_raster_info(
                    base_raster_task_list[0])['pixel_size'], 'intersection'),
            target_path_list=aligned_path_list,
            task_name='align base rasters for %s' % os.path.basename(
                target_10s_path))
        dependent_task_list.append(align_task)
        path_list = [(path, 1) for path in aligned_path_list]
    target_nodata = -9999.

    add_raster_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(target_nodata, 'raw')] + path_list, add_op, target_10s_path,
            gdal.GDT_Float32, target_nodata),
        target_path_list=[target_10s_path],
        dependent_task_list=dependent_task_list,
        task_name=f'add rasters {os.path.basename(target_10s_path)}')

    return schedule_aggregate_to_degree(
        task_graph, target_10s_path, aggregate_func, add_raster_task)


def schedule_aggregate_to_degree(
        task_graph, base_raster_path, aggregate_func, base_raster_task,
        target_raster_path_name=None):
    """Schedule an aggregate of 1D approximation."""
    if not target_raster_path_name:
        one_degree_raster_path = (
            base_raster_path.replace('_10s_', '_1d_').replace(
                '_30s_', '_1d_'))
    else:
        one_degree_raster_path = target_raster_path_name
    one_degree_task = task_graph.add_task(
        func=aggregate_to_degree,
        args=(base_raster_path, aggregate_func, one_degree_raster_path),
        target_path_list=[one_degree_raster_path],
        dependent_task_list=[base_raster_task],
        task_name=f'to degree {os.path.basename(one_degree_raster_path)}')
    return (one_degree_task, one_degree_raster_path)


def aggregate_to_degree(raster_path, aggregate_func, target_path):
    """Aggregate input raster to a degree.

    Parameters:
        base_raster_path (string): path to a WGS84 projected raster.
        target_path (string): path to desired target raster that will be
            an aggregated version of `base_raster_path` by the function
            `aggregate_func`.

    Returns:
        None.

    """
    base_raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    base_gt = base_raster.GetGeoTransform()
    base_band = base_raster.GetRasterBand(1)
    base_nodata = base_band.GetNoDataValue()

    wgs84sr = osr.SpatialReference()
    wgs84sr.ImportFromEPSG(4326)

    driver = gdal.GetDriverByName('GTiff')
    n_rows = int(
        abs((base_gt[5] * base_band.YSize) / 1.0))
    n_cols = int(
        abs((base_gt[1] * base_band.XSize) / -1.0))
    target_raster = driver.Create(
        target_path, n_cols, n_rows, 1, gdal.GDT_Float32)
    target_raster.SetProjection(wgs84sr.ExportToWkt())
    degree_geotransform = [base_gt[0], 1., 0., base_gt[3], 0., -1.]
    target_raster.SetGeoTransform(degree_geotransform)
    target_band = target_raster.GetRasterBand(1)
    target_band.SetNoDataValue(base_nodata)
    target_band.Fill(base_nodata)

    base_y_winsize = int(round(abs(1. / base_gt[5])))
    base_x_winsize = int(round(abs(1. / base_gt[1])))

    last_time = time.time()
    for row_index in range(n_rows):
        lat_coord = (
            degree_geotransform[3] + degree_geotransform[5] * row_index)
        base_y_coord = int((lat_coord - base_gt[3]) / base_gt[5])
        target_y_coord = int(
            (lat_coord - degree_geotransform[3]) / degree_geotransform[5])
        for col_index in range(n_cols):
            long_coord = (
                degree_geotransform[0] + degree_geotransform[1] * col_index)
            base_x_coord = int((long_coord - base_gt[0]) / base_gt[1])
            target_x_coord = int(
                (long_coord - degree_geotransform[0]) / degree_geotransform[1])

            base_array = base_band.ReadAsArray(
                xoff=base_x_coord, yoff=base_y_coord,
                win_xsize=base_x_winsize, win_ysize=base_y_winsize)
            valid_array = ~numpy.isclose(base_array, base_nodata)
            if valid_array.any():
                target_band.WriteArray(
                    numpy.array([[aggregate_func(base_array[valid_array])]]),
                    xoff=target_x_coord, yoff=target_y_coord)

            current_time = time.time()
            if (current_time - last_time) > 5.0:
                LOGGER.info(
                    "%.2f%% complete", 100.0 * float(row_index+1) / n_rows)
                last_time = current_time
    target_band.FlushCache()
    target_band = None
    target_raster = None
    base_band = None
    base_raster = None
    LOGGER.info("100%% complete")


def sum_num_sum_denom(
        num1_array, num2_array, denom1_array, denom2_array, nodata):
    """Calculate sum of num divided by sum of denom."""
    result = numpy.empty_like(num1_array)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(num1_array, nodata) &
        ~numpy.isclose(num2_array, nodata) &
        ~numpy.isclose(denom1_array, nodata) &
        ~numpy.isclose(denom2_array, nodata))
    result[valid_mask] = (
        num1_array[valid_mask] + num2_array[valid_mask]) / (
        denom1_array[valid_mask] + denom2_array[valid_mask] + 1e-9)
    return result


def avg_3_op(array_1, array_2, array_3, nodata):
    """Average 3 arrays. Skip nodata."""
    result = numpy.empty_like(array_1)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(array_1, nodata) &
        ~numpy.isclose(array_2, nodata) &
        ~numpy.isclose(array_3, nodata))
    result[valid_mask] = (
        array_1[valid_mask] +
        array_2[valid_mask] +
        array_3[valid_mask]) / 3.
    return result


def weighted_avg_3_op(
        array_1, array_2, array_3,
        scalar_1, scalar_2, scalar_3,
        nodata):
    """Weighted average 3 arrays. Skip nodata."""
    result = numpy.empty_like(array_1)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(array_1, nodata) &
        ~numpy.isclose(array_2, nodata) &
        ~numpy.isclose(array_3, nodata))
    result[valid_mask] = (
        array_1[valid_mask]/scalar_1 +
        array_2[valid_mask]/scalar_2 +
        array_3[valid_mask]/scalar_3) / 3.
    return result


def count_ge_one(array):
    """Return count of elements >= 1."""
    return numpy.count_nonzero(array >= 1)


def prop_diff_op(array_a, array_b, nodata):
    """Calculate prop change from a to b."""
    result = numpy.empty_like(array_a)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(array_a, nodata) &
        ~numpy.isclose(array_b, nodata))
    # the 1e-12 is to prevent a divide by 0 error
    result[valid_mask] = (
        array_b[valid_mask] - array_a[valid_mask]) / (
            array_a[valid_mask] + 1e-12)
    return result


def calc_relevant_pop(
        base_pop_path,
        prod_poll_indep_en_path, nut_req_en_path,
        prod_poll_indep_fo_path, nut_req_fo_path,
        prod_poll_indep_va_path, nut_req_va_path,
        logic_ufunc, target_raster_path):
    """This function is meant to duplicate the SQL query that Becky wrote:

    ALTER TABLE relevant_population ADD relevant_pop_cur INTEGER;
    ALTER TABLE relevant_population ADD relevant_pop_ssp1 INTEGER;
    ALTER TABLE relevant_population ADD relevant_pop_ssp3 INTEGER;
    ALTER TABLE relevant_population ADD relevant_pop_ssp5 INTEGER;
    UPDATE relevant_population SET relevant_pop_cur = cur;
    UPDATE relevant_population SET relevant_pop_ssp1 = gpw_v4_e_atot_pop_30s_ssp1;
    UPDATE relevant_population SET relevant_pop_ssp3 = gpw_v4_e_atot_pop_30s_ssp3;
    UPDATE relevant_population SET relevant_pop_ssp5 = gpw_v4_e_atot_pop_30s_ssp5;
    UPDATE relevant_population SET relevant_pop_cur = 0 WHERE prod_poll_indep_en_cur > nut_req_en_1d_cur AND prod_poll_indep_fo_cur > nut_req_fo_1d_cur AND prod_poll_indep_va_cur > nut_req_va_1d_cur;
    UPDATE relevant_population SET relevant_pop_ssp1 = 0 WHERE prod_poll_indep_en_ssp1 > nut_req_en_1d_ssp1 AND prod_poll_indep_fo_ssp1 > nut_req_fo_1d_ssp1 AND prod_poll_indep_va_ssp1 > nut_req_va_1d_ssp1;
    UPDATE relevant_population SET relevant_pop_ssp3 = 0 WHERE prod_poll_indep_en_ssp3 > nut_req_en_1d_ssp3 AND prod_poll_indep_fo_ssp3 > nut_req_fo_1d_ssp3 AND prod_poll_indep_va_ssp3 > nut_req_va_1d_ssp3;
    UPDATE relevant_population SET relevant_pop_ssp5 = 0 WHERE prod_poll_indep_en_ssp5 > nut_req_en_1d_ssp5 AND prod_poll_indep_fo_ssp5 > nut_req_fo_1d_ssp5 AND prod_poll_indep_va_ssp5 > nut_req_va_1d_ssp5;

    """

    def relevant_pop_op(
            base_pop_array,
            prod_poll_indep_en_array, nut_req_en_array,
            prod_poll_indep_fo_array, nut_req_fo_array,
            prod_poll_indep_va_array, nut_req_va_array):
        result = numpy.copy(base_pop_array)
        result[
            logic_ufunc(
                (prod_poll_indep_en_array > nut_req_en_array),
                logic_ufunc(
                    (prod_poll_indep_fo_array > nut_req_fo_array),
                    (prod_poll_indep_va_array > nut_req_va_array)))] = 0.0
        return result

    base_info = pygeoprocessing.get_raster_info(base_pop_path)

    pygeoprocessing.raster_calculator(
        [(base_pop_path, 1),
         (prod_poll_indep_en_path, 1), (nut_req_en_path, 1),
         (prod_poll_indep_fo_path, 1), (nut_req_fo_path, 1),
         (prod_poll_indep_va_path, 1), (nut_req_va_path, 1)],
        relevant_pop_op, target_raster_path,
        base_info['datatype'], base_info['nodata'][0])

if __name__ == '__main__':
    main()
