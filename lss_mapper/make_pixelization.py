from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table,vstack
import numpy as np
from .flatmaps import FlatMapInfo
from .map_utils import createCountsMap, createMeanStdMaps, createMask, removeDisconnected
from .estDepth import get_depth
from astropy.io import fits

class MaxPixs(PipelineStage) :
    name="MaxPixs"
    inputs=[('raw_data', None)]
    outputs=[('flatmap_info', FitsFile)]
    config_options = {'res':0.0285, 'pad':0.1, 'band':'i',
                    'flat_project':'CAR', 'mask_type':'sirius'}
    bands=['g','r','i','z','y']

    def run(self) :
        """
        Main function.
        This stage:
        - Reduces the raw catalog by imposing quality cuts, a cut on i-band magnitude and a star-galaxy separation cat.
        - Produces mask maps, dust maps, depth maps and star density maps.
        """
        band=self.config['band']

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

        if band not in self.bands :
            raise ValueError("Band "+band+" not available")

        print('Initial catalog size: %d'%(len(cat)))
            
        # Clean nulls and nans
        print("Basic cleanup")
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
        print("Will drop %d rows"%(len(sel)-np.sum(sel)))
        cat.remove_columns(isnull_names)
        cat.remove_rows(~sel)

        fsk=FlatMapInfo.from_coords(cat['ra'],cat['dec'],self.config['res'],
                                    pad=self.config['pad']/self.config['res'],
                                    projection=self.config['flat_project'])

        ####
        # Generate flatmap info
        flatmap_info_descr = 'FlatmapInfo'
        fsk.write_flat_map(self.get_output('flatmap_info'), np.ones(fsk.npix), descript=flatmap_info_descr)

if __name__ == '__main__':
    cls = PipelineStage.main()
