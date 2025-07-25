ens=cC211.060.80

# scp -r /nvme/h/cy22yl1/projectData/02_discNJN_1D/${ens}/data_post/ li47@juwels-booster.fz-juelich.de:/p/project/ngff/li47/code/projectData/02_discNJN_1D/${ens}/loop_cyclone/
rsync -av --progress /nvme/h/cy22yl1/projectData/02_discNJN_1D/${ens}/data_post/ li47@juwels-booster.fz-juelich.de:/p/project/ngff/li47/code/projectData/02_discNJN_1D/${ens}/loop_cyclone_scalingCheck/