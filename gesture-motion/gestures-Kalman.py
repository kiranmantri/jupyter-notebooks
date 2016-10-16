import os
import fnmatch

import pandas as pd
import numpy as np
import sympy as sp
import cv2
from MarkerFileParser import MarkerFileParser

from sklearn import preprocessing
from sklearn.decomposition import PCA

from tabulate import tabulate

pd.set_option('display.multi_sparse', False)
create_graphs = False
mkr = None

allFiles= []


# ----- First get all file names from the data dir ------ #
print "Scaning Files..."
for root, dirnames, filenames in os.walk('gestures_clean'):
    for filename in fnmatch.filter(filenames, '*.mkr'):
        allFiles.append(os.path.join(root, filename))

print len(allFiles), " found !"
# ------------------------------------------------------- #


# --- For each file, we must parse the contents,
# --- Get Hand information
# --- import to Pandas structure
# --- create appropriate headers for X,Y,X (for each hand)
# --- Most of this job is done in the MarkerFileParser class

for num, file in enumerate(allFiles[:]):
    print num, " Parsing: ", file ,
    hdf5Path = os.path.join('output',os.path.basename(file)+'.h5')
    figPath = os.path.join('figs',os.path.basename(file)+'.png')
    movName = os.path.join('figs',os.path.basename(file)+'.mp4')

    mkr = MarkerFileParser(file)

    if not (mkr._status):
        print "(this is fucked up !)"
        continue
    else:
        print ""


    observed_df = pd.DataFrame()
    estimated_df = pd.DataFrame()
    corrected_df = pd.DataFrame()

    d = mkr.getDataSet()
    hands = mkr.getHands()

    for h in sorted(hands):
        markers = mkr.getMarkers(hand=h)
        for m in markers:
            c = d.xs([h,m], level=['hand','markerName'], axis=1).drop('v', axis=1)

            dt = mkr.getDeltaT()

            kf = cv2.KalmanFilter(dynamParams=6, measureParams=3, controlParams=0, type=cv2.CV_64FC1)

            kf.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0],
                                            [0, 1, 0, 0, dt, 0],
                                            [0, 0, 1, 0, 0, dt],
                                            [0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 1]
                                           ],dtype="float64")

            kf.processNoiseCov = np.array([[(dt**4)/4, 0, 0, (dt**3)/2, 0, 0],
                                          [0, (dt**4)/4 , 0, 0, (dt**3)/2, 0],
                                          [0, 0, (dt**4)/4, 0, 0, (dt**3)/2],
                                          [(dt**3)/2, 0, 0, (dt**2), 0, 0],
                                          [0, (dt**3)/2, 0, 0, (dt**2), 0],
                                          [0, 0, (dt**3)/2, 0, 0, (dt**2)]
                                 ], dtype="float64")


            kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0, 0]
                                            ],dtype="float64")

            kf.measurementNoiseCov = np.array([[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1,]
                                            ],dtype="float64")*1E-9


            observed = []
            estimated = []
            corrected = []

            for index, row in c.iterrows():
                if index==0 :
                    kf.statePost[0,0] = row.values[0]
                    kf.statePost[1,0] = row.values[1]
                    kf.statePost[2,0] = row.values[2]

                    kf.statePre[0,0] = row.values[0]
                    kf.statePre[1,0] = row.values[1]
                    kf.statePre[2,0] = row.values[2]

                measurement = row.values[0:3].reshape(3)

                predicted = kf.predict()[0:3].reshape(3)
                corrected.append(kf.correct(measurement)[0:3].reshape(3))

                if index==0 :
                    estimated.append(row.values[0:3])
                else:
                    estimated.append(predicted)

                observed.append(measurement)


            tuples= [ (cn, h, m) for cn in ['x','y','z'] ]
            fields = pd.MultiIndex.from_tuples(tuples, names=['cartesian','hand', 'markerName'])

            #print fields
            estimated = pd.DataFrame(data=np.array(estimated), columns=fields)
            observed = pd.DataFrame(data=np.array(observed), columns=fields)
            corrected = pd.DataFrame(data=np.array(corrected), columns=fields)

            estimated_df = pd.concat( [ estimated_df, estimated], axis=1)
            observed_df = pd.concat( [ observed_df, observed], axis=1)



    ######### Finished Processing.... now storing to the hdf5 file ############
    print "-"*40
    pca = PCA(n_components=50, whiten=True)
    pca.fit(estimated_df.values)
    estimated_pca_df = pd.DataFrame(data=pca.components_)

    pca = PCA(n_components=50, whiten=True)
    pca.fit(observed_df.values)
    observed_pca_df = pd.DataFrame(data=pca.components_)


    # store = pd.HDFStore(hdf5Path)
    # store['timestep'] = pd.DataFrame(data=np.array([[mkr.getDeltaT()]]), columns=['timestep'])
    # store['estimated'] = estimated_df
    # store['observed'] = observed_df
    #
    # store["estimated_pca"] = estimated_pca_df
    # store["observed_pca"] = observed_pca_df
    #
    # store.close()

    continue

    if create_graphs:
        # #print estimated_df.head()
        import matplotlib.pyplot as plt
        from pandas.tools.plotting import parallel_coordinates
        from pandas.tools.plotting import andrews_curves

        fig = plt.figure(figsize=(15,4))

        estimated_df.xs(['x','left'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='X Estimated Left hand', color='r', ls='-')
        observed_df.xs(['x','left'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='X Observed Left hand', color='r', ls='--')

        estimated_df.xs(['y','left'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Y Estimated Left hand', color='g', ls='-')
        observed_df.xs(['y','left'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Y Observed Left hand', color='g', ls='--')

        estimated_df.xs(['z','left'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Z Estimated Left hand', color='b', ls='-')
        observed_df.xs(['z','left'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Z Observed Left hand', color='b', ls='--')

        estimated_df.xs(['x','right'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='X Estimated Right hand', color='r', ls='-')
        observed_df.xs(['x','right'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='X Observed Right hand', color='r', ls='--')

        estimated_df.xs(['y','right'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Y Estimated Right hand', color='g', ls='-')
        observed_df.xs(['y','right'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Y Observed Right hand', color='g', ls='--')

        estimated_df.xs(['z','right'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Z Estimated Right hand', color='b', ls='-')
        observed_df.xs(['z','right'], level=['cartesian','hand'], axis=1).mean(axis=1).plot(label='Z Observed Right hand', color='b', ls='--')

        # plt.plot(observed_df['x'].values,"r-",)
        # plt.plot(estimated_df['x'].values,"r--")
        #
        # plt.plot(observed_df['y'].values,"g-", )
        # plt.plot(estimated_df['y'].values,"g--")
        #
        # plt.plot(observed_df['z'].values,"b-")
        # plt.plot(estimated_df['z'].values,"b--")

        plt.xlim(xmax=len(estimated)*(1.2))
        plt.title(file)
        plt.legend(fontsize=8)
        figPath = os.path.join('figs',os.path.basename(file)+'.png')
        #plt.savefig(figPath, dpi=300)
        plt.show()
        plt.close()


        from matplotlib import pyplot as plt
        import pylab
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        fig = pylab.figure()
        ax = Axes3D(fig)

        eL =  estimated_df.xs(['left'], level=['hand'], axis=1)
        XL = eL.xs('x',level='cartesian', axis=1).mean(axis=1).values
        YL = eL.xs('y',level='cartesian', axis=1).mean(axis=1).values
        ZL = eL.xs('z',level='cartesian', axis=1).mean(axis=1).values

        eR = estimated_df.xs(['right'], level=['hand'], axis=1)
        XR = eR.xs('x',level='cartesian', axis=1).mean(axis=1).values
        YR = eR.xs('y',level='cartesian', axis=1).mean(axis=1).values
        ZR = eR.xs('z',level='cartesian', axis=1).mean(axis=1).values


        oL =  observed_df.xs(['left'], level=['hand'], axis=1)
        oXL = oL.xs('x',level='cartesian', axis=1).mean(axis=1).values
        oYL = oL.xs('y',level='cartesian', axis=1).mean(axis=1).values
        oZL = oL.xs('z',level='cartesian', axis=1).mean(axis=1).values

        oR =  observed_df.xs(['right'], level=['hand'], axis=1)
        oXR = oR.xs('x',level='cartesian', axis=1).mean(axis=1).values
        oYR = oR.xs('y',level='cartesian', axis=1).mean(axis=1).values
        oZR = oR.xs('z',level='cartesian', axis=1).mean(axis=1).values



        ax.set_xlim(xmax=min(min(XR),min(XL)), xmin=max(max(XR),max(XL)))
        ax.set_ylim(ymax=min(min(YR),min(YL)), ymin=max(max(YR),max(YL)))
        ax.set_zlim(zmax=min(min(ZR),min(ZL)), zmin=max(max(ZR),max(ZL)))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.invert_yaxis()
        ax.invert_zaxis()


        def init():
            line = ax.scatter(XL[0],YL[0],ZL[0], marker='*', color='g')
            line = ax.scatter(XR[0],YR[0],ZR[0], marker='*', color='r')

            line = ax.scatter(oXL[0],oYL[0],oZL[0], marker='o', color='g')
            line = ax.scatter(oXR[0],oYR[0],oZR[0], marker='o', color='r')

        def animate(i):
            line = ax.scatter(XL[i],YL[i],ZL[i], marker='*', color='g')
            line = ax.scatter(XR[i],YR[i],ZR[i], marker='*', color='r')

            line = ax.scatter(oXL[i],oYL[i],oZL[i], marker='o', color='g')
            line = ax.scatter(oXR[i],oYR[i],oZR[i], marker='o', color='r')

        lines_ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(XL), repeat=False, interval=1, blit=False)


        print "Writing:", movName, " . . ."
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        lines_ani.save(movName, writer=writer)
        plt.show()
        plt.close()







print "done..."
