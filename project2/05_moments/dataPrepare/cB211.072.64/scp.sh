ens=cB211.072.64

# hostname=li47@login.jupiter.fz-juelich.de
hostname=li47@juwels-booster.fz-juelich.de

rsync -av --progress /nvme/h/cy22yl1/projectData/05_moments/${ens}/data_merge/ ${hostname}:/p/project1/ngff/li47/code/projectData/05_moments/${ens}/cyclone/