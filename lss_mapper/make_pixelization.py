from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table,vstack
import numpy as np
from .flatmaps import FlatMapInfo
from .map_utils import createCountsMap, createMeanStdMaps, createMask, removeDisconnected
from .estDepth import get_depth
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MakePixs(PipelineStage) :
    name = "MakePixs"
    inputs = [('raw_data', None)]
    outputs = [('clean_catalog',FitsFile), ('flatmap_info', FitsFile)]
    config_options = {'res':0.0285, 'pad':0.1,
                      'flat_project':'CAR',
                      'shift_to_equator':False}

    def run(self) :
        """
        Main function.
        This stage:
        - Reduces the raw catalog by imposing quality cuts, a cut on i-band magnitude and a star-galaxy separation cat.
        - Produces mask maps, dust maps, depth maps and star density maps.
        """

        #Read list of files
        f=open(self.get_input('raw_data'))
        files=[s.strip() for s in f.readlines()]
        f.close()

        #Read catalog
        cat=Table.read(files[0])
        if len(cat)>1 :
            for fname in files[1:] :
                c=Table.read(fname)
                cat=vstack([cat,c],join_type='exact')

        logger.info('Initial catalog size: %d'%(len(cat)))
            
        # Clean nulls and nans
        logger.info("Applying basic quality cuts.")
        sel=np.ones(len(cat),dtype=bool)
        names=[n for n in cat.keys()]
        isnull_names=[]
        for key in cat.keys() :
            if key.__contains__('isnull') :
                sel[cat[key]]=0
                isnull_names.append(key)
            else:
                if not key.startswith("pz_") : #Keep photo-z's even if they're NaNs
                    sel[np.isnan(cat[key])]=0
                logger.info("Will drop %d rows"%(len(sel)-np.sum(sel)))
        cat.remove_columns(isnull_names)
        cat.remove_rows(~sel)

        fsk=FlatMapInfo.from_coords(cat['ra'],cat['dec'],self.config['res'],
                                    pad=self.config['pad']/self.config['res'],
                                    projection=self.config['flat_project'],
                                    move_equator=self.config['shift_to_equator'])

        ####
        # Generate flatmap info
        flatmap_info_descr = 'FlatmapInfo'
        fsk.write_flat_map(self.get_output('flatmap_info'), np.ones(fsk.npix),
                           descript=flatmap_info_descr)

        ####
        # Write final catalog
        # 1- header
        logger.info("Writing cleaned catalog.")
        hdr=fits.Header()
        prm_hdu=fits.PrimaryHDU(header=hdr)
        # 2- Catalog
        cat_hdu=fits.table_to_hdu(cat)
        # 3- Actual writing
        hdul=fits.HDUList([prm_hdu,cat_hdu])
        hdul.writeto(self.get_output('clean_catalog'), overwrite=True)
        ####

if __name__ == '__main__':
    cls = PipelineStage.main()
