import numpy as np
import xarray as xray
import netCDF4
import warnings
import sys
from warnings import warn
from scipy import linalg as lin
from scipy import signal as sig
from scipy import fftpack as fft
from scipy import interpolate as naiso
import gsw

class griddedFile(object):
    
    def __init__(self, fname):
        """Wrapper for NASA satellite netCDF files
            on a lat-lon grid
        """
        # self.nc = netCDF4.Dataset(fname)
        self.nc = xray.open_dataset(fname, engine='pynio')
        # self.Ny, self.Nx = self.nc[areaname].shape  
        # self.Nt, self.Ny, self.Nx = self.nc[maskname].shape  

    def spectrum_2d(self, varname='analysed_sst', lonname='lon', latname='lat', maskname='mask', filename=False, grady=False, lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, nbins=32, MAX_LAND=0.01):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
            in the box defined by lonrange and latrange.
        """
        #tlon = np.roll(np.ma.masked_array(self.nc.variables[lonname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = np.roll(np.ma.masked_array(self.nc.variables[latname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlon = self.nc[lonname].sel(lon=slice(lonrange[0],
                                          lonrange[1])).values
        tlat = self.nc[latname].sel(lat=slice(latrange[0],
                                          latrange[1])).values
        #tlon = xray.DataArray(np.ma.masked_array(self.nc[lonname][:],
                                                 #mask)).roll( nlon=roll )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = xray.DataArray(np.ma.masked_array(self.nc[latname][:],
                                                 #mask)).roll( nlon=roll )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        # tlon[tlon<0.] += 360.

        # step 1: figure out the box indices
        #lon, lat = np.meshgrid(tlon, tlat)
        Nx = len(tlon)
        Ny = len(tlat)
        #########
        # derive dx, dy using the gsw package (takes a long time)
        #########
        #dx = np.zeros((Ny, Nx))
        #dy = np.zeros_like(dx)
        #for j in range(Ny-1):
        #    for i in range(Nx-1):
        #        dx[j,i] = gsw.distance([tlon[i],tlon[i+1]], [tlat[j],tlat[j]])
        #        dy[j,i] = gsw.distance([tlon[i],tlon[i]], [tlat[j],tlat[j+1]])
        #########
        # derive dx, dy just at the center point
        #########
        a = gsw.earth_radius
        dx = a * np.cos(np.pi/180.*tlat[Ny/2]) * np.pi/180.*np.diff(tlon)[Nx/2]
        dy = a * np.pi/180.*np.diff(tlat)[Ny/2]

        # step 2: load the data
        T = self.nc[varname].sel(lon=slice(lonrange[0], lonrange[1]), 
                                                          lat=slice(latrange[0], latrange[1])).values

        # step 3: figure out if there is too much land in the box
        #MAX_LAND = 0.01 # only allow up to 1% of land
        #mask_domain = mask.roll( nlon=roll )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        region_mask = self.nc[maskname].sel(lon=slice(lonrange[0], lonrange[1]), 
                                                          lat=slice(latrange[0], latrange[1])).values - 1.
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction == 0.:
            # no problem
            pass
        elif land_fraction <= MAX_LAND:
            crit = 'false'
            errstr = 'The sector has too much land. land_fraction = ' + str(land_fraction)
            warnings.warn(errstr)
            #raise ValueError('The sector has too much land. land_fraction = ' + str(land_fraction))
        else:    
            # do some interpolation
            errstr = 'The sector has land (land_fraction=%g) but we are interpolating it out.' % land_fraction
            warnings.warn(errstr)
        
        # step 4: figure out FFT parameters (k, l, etc.) and set up result variable
        #dlon = lon[np.round(np.floor(lon.shape[0]*0.5)), np.round(
        #             np.floor(lon.shape[1]*0.5))+1]-lon[np.round(
        #             np.floor(lon.shape[0]*0.5)), np.round(np.floor(lon.shape[1]*0.5))]
        #dlat = lat[np.round(np.floor(lat.shape[0]*0.5))+1, np.round(
        #             np.floor(lat.shape[1]*0.5))]-lat[np.round(
        #             np.floor(lat.shape[0]*0.5)), np.round(np.floor(lat.shape[1]*0.5))]

        # Spatial step
        #dx = gfd.A*np.cos(np.radians(lat[np.round(
        #             np.floor(lat.shape[0]*0.5)),np.round(
        #             np.floor(lat.shape[1]*0.5))]))*np.radians(dlon)
        #dy = gfd.A*np.radians(dlat)
        
        # Wavenumber step
        #dx_domain = dx[jmin:jmax,imin:imax].copy()
        #dy_domain = dy[jmin:jmax,imin:imax].copy()
        #dk = np.diff(k)[0]*.5/np.pi
        #dl = np.diff(l)[0]*.5/np.pi
        k = fft.fftshift(fft.fftfreq(Nx, dx))
        l = fft.fftshift(fft.fftfreq(Ny, dy))
        dk = np.diff(k)[0]
        dl = np.diff(l)[0]

        ################################
        ###  MUR data is given daily individually ###
        ################################
        #Nt = T.shape[0]
        #Decor_lag = 13
        #tilde2_sum = np.zeros((Ny, Nx))
        #Ti2_sum = np.zeros((Ny, Nx))
        #Days = np.arange(0,Nt,Decor_lag)
        #Neff = len(Days)
        #for n in Days:
        Ti = np.ma.masked_array(T.copy(), region_mask)
            
        # step 5: interpolate the missing data (only if necessary)
        if land_fraction>0. and land_fraction<MAX_LAND:
            Ti = interpolate_2d(Ti)
        elif land_fraction==0.:
            # no problem
            pass
        else:
            sys.exit(0)
        
        # step 6: detrend the data in two dimensions (least squares plane fit)
        Ti -= trend_2d(Ti)

        # step 7: window the data
        # Hanning window
        windowx = sig.hann(Nx)
        windowy = sig.hann(Ny)
        window = windowx*windowy[:,np.newaxis] 
        Ti *= window
            
        Ti2 = Ti**2

        # step 8: do the FFT for each timestep and aggregate the results
        Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
        tilde2 = np.real(Tif*np.conj(Tif))

        # step 9: check whether the Plancherel theorem is satisfied
        breve2 = tilde2/((Nx*Ny)**2*dk*dl)
        if land_fraction == 1.:
            #np.testing.assert_almost_equal(breve2_ave.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
            np.testing.assert_almost_equal( breve2.sum() / ( dx * dy * Ti2.sum() ), 1., decimal=5)
            
        # step 10: derive the isotropic spectrum
        kk, ll = np.meshgrid(k, l)
        K = np.sqrt( kk**2 + ll**2 )
        #Ki = np.linspace(0, k.max(), nbins)
        if k.max() > l.max():
            Ki = np.linspace(0, l.max(), nbins)
        else:
            Ki = np.linspace(0, k.max(), nbins)
        #Ki = np.linspace(0, K.max(), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        invalid = Kidx[-1]
        area = np.bincount(Kidx)
        #PREVIOUS: isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=breve2_sum.ravel()) / area )[:-1] *Ki*2.*np.pi**2
        isotropic_PSD = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=breve2.ravel()) / area )[:-1] * Ki
        
        # Usage of digitize
        #>>> x = np.array([-0.2, 6.4, 3.0, 1.6, 20.])
        #>>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        #>>> inds = np.digitize(x, bins)
        #array([0, 4, 3, 2, 5])
        
        # Usage of bincount 
        #>>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
        #array([1, 3, 1, 1, 0, 0, 0, 1])
        # With the option weight
        #>>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
        #>>> x = np.array([0, 1, 1, 2, 2, 2])
        #>>> np.bincount(x,  weights=w)
        #array([ 0.3,  0.7,  1.1])  <- [0.3, 0.5+0.2, 0.7+1.-0.6]
        
        # step 10: return the results
        return Nx, Ny, k, l, Ti2, tilde2, breve2, Ki[:], isotropic_PSD[:], area[1:-1], tlon, tlat
    
    def structure_function(self, varname='sst', maskname='hdf_file', qualname='qual', lonrg=(154.9,171.7), latrg=(30,45.4), q=2, rand=10000, MAX_GAP=0.5, detre=True, windw=True, iso=False, lin_near=True):
        """Calculate a structure function of Matlab variable 'varname'
           in the box defined by lonrange and latrange.
        """
        nc_qual = xray.open_dataset(maskname, engine='pynio')
        quality = nc_qual[qualname]
        lon = self.nc[varname].sel(lat=slice(latrg[1], latrg[0]), 
                                   lon=slice(lonrg[0], lonrg[1])).where(quality>3.).lon
        lat = self.nc[varname].sel(lat=slice(latrg[1], latrg[0]), 
                                   lon=slice(lonrg[0], lonrg[1])).where(quality>3.).lat

        ############
        # load data
        ############
        T = self.nc[varname].sel(lat=slice(latrg[1], latrg[0]), 
                                   lon=slice(lonrg[0], lonrg[1])).where(quality>3.).values
        
        # define variables
        Ny, Nx = T.shape
        n = np.arange(0, np.log2(3*Nx/4.), dtype='i4')
        ndel = len(n)
        # Spatial step
        dx = gsw.earth.distance([lon[int(Nx/2)], lon[int(Nx/2)+1]], [lat[int(Ny/2)], lat[int(Ny/2)]])
        dy = gsw.earth.distance([lon[int(Nx/2)], lon[int(Nx/2)]], [lat[int(Ny/2)], lat[int(Ny/2)+1]])
        # dx = gsw.earth.distance([.5*(lonrg[0]+lonrg[1])-.5, .5*(lonrg[0]+lonrg[1])+.5], 
        #                        [.5*(latrg[0]+latrg[1]), .5*(latrg[0]+latrg[1])])
        # dy = gsw.earth.distance([.5*(lonrg[0]+lonrg[1]), .5*(lonrg[0]+lonrg[1])], 
        #                       [.5*(latrg[0]+latrg[1])-.5, .5*(latrg[0]+latrg[1])+.5])
        
        ############
        # Figure out if there is too much gaps in the box
        # MAX_GAP = 0.5 (only allow up to 50% of gap)
        ############
        mask_domain = np.ma.masked_invalid(T).mask
        gap_fraction = mask_domain.sum().astype('f8') / (Ny*Nx)
        # step 5: If gap_fraction is larger than 50%, give an error
        if gap_fraction > MAX_GAP:
            # crit = 'false'
            # errstr = 'The sector has too much land or mask. masked_fraction = ' + str(gap_fraction)
            # warnings.warn(errstr)
            # sys.exit(0)
            # raise ValueError('The sector has too much land or mask. masked_fraction = ' + str(gap_fraction))
            return None
        else:
            # no problem
            # step 6: detrend the data in two dimensions (least squares plane fit)
            Ti = np.ma.masked_invalid(T.copy())
            if lin_near:
                interpTi = interpolate_2d(np.ma.masked_invalid(T.copy()))
                interpTi = interpolate_2d(np.ma.masked_invalid(interpTi), 
                                          meth='nearest')
            else:
                interpTi = interpolate_2d(np.ma.masked_invalid(T.copy()), 
                                          meth='nearest')
            trendTi = trend_2d(interpTi)
            Ti -= trendTi

            # window the data
            # Hanning window
            if windw:
                windowx = sig.hann(Nx)
                windowy = sig.hann(Ny)
                window = windowx*windowy[:,np.newaxis]
                Ti *= window

            if iso:
                ##################
                # Calculate structure functions isotropically
                # 
                # Due to difference between meridional and zonal distance
                # the length scale will only be defined as the grid points
                # in between the two points
                ##################
                L = 2**n
                S = np.empty(ndel)
                S[:] = np.nan
                #dT = np.zeros(rand)
                dT = 0.
                for m in range(ndel):
                    for n in range(rand):
                        i = np.random.randint(0, Nx)
                        j = np.random.randint(0, Ny)
                        angle = 2.*np.pi * np.random.uniform(0., 1.)
                        r = 2**m
                        di = r*np.cos(angle)
                        dj = r*np.sin(angle)
                        i2 = round(i + di)
                        j2 = round(j + dj)
                        if i2 >= Nx or i2 < 0:
                            i2 = round(i - di)
                            #i2 -= Nx
                        if j2 >= Ny or j2 < 0:
                            j2 = round(j - dj)
                            #j2 -= Ny

                        #dT[n] = np.abs( Ti[j,i] - Ti[j2,i2] )**q
                        dT += np.abs( Ti[j,i] - Ti[j2,i2] )**q
                    #H[m] = dT.mean()
                    S[m] = dT/rand

                return dx, dy, L, S, lon, lat, gap_fraction

            else:
                ##################
                # Calculate structure functions along each x-y axis
                ##################
                Li = 2.**n * dx
                Lj = 2.**n * dy
                Si = np.empty(ndel)
                Si[:] = np.nan
                Sj = Si.copy()
                for m in range(ndel):
                    dSSTi = np.abs(Ti[:, 2**m:] - Ti[:, :-2**m])**q
                    dSSTj = np.abs(Ti[2**m:] - Ti[:-2**m])**q

                    Si[m] = np.nanmean(dSSTi)
                    Sj[m] = np.nanmean(dSSTj)                  

                return dx, dy, Li, Lj, Si, Sj, lon, lat, gap_fraction

def interpolate_2d(Ti, meth='linear'):
    """Interpolate a 2D field
    """
    Ny, Nx = Ti.shape
    x = np.arange(0,Nx)
    y = np.arange(0,Ny)
    X,Y = np.meshgrid(x,y)
    Zr = Ti.ravel()
    Xr = np.ma.masked_array(X.ravel(), Zr.mask)
    Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
    Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
    Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
    Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T,
                                    Zr.compressed(), np.array([Xm,Ym]).T, method=meth)
    Znew = Zr.data
    Znew[Zr.mask] = Zm
    Znew.shape = Ti.shape

    return Znew
 
def trend_2d(Ti):
    """Give Linear plane fit of a 2D field
    """
    Ny, Nx = Ti.shape
    d_obs = np.reshape(Ti, (Nx*Ny,1))
    G = np.ones((Ny*Nx,3))
    for i in range(Ny):
        G[Nx*i:Nx*i+Nx, 0] = i+1
        G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
    m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)
    Lin_trend = np.reshape(d_est, (Ny, Nx))
    
    return Lin_trend