# docker build -t insilica/biobricks:3.0-cuda .
# you need nvidia-docker2 to run this container 
#    see the install instructions 
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
FROM rocker/cuda:cuda11.1

RUN apt-get update -y
RUN apt-get -y --no-install-recommends install git libxml2-dev

RUN apt-get install -y python3 python3-pip
RUN pip install dvc
RUN pip install dvc-s3

# Biobricks dependencies
RUN install2.r --error arrow
RUN install2.r --error dplyr
RUN install2.r --error fs
RUN install2.r --error purrr
RUN install2.r --error xml2
RUN install2.r --error readr
RUN install2.r --error rvest
RUN install2.r --error DBI
RUN install2.r --error RSQLite
RUN install2.r --error yaml

# Biobricks testing
RUN install2.r --error remotes
RUN Rscript -e "remotes::install_github('biobricks-ai/biobricks-r')"
RUN Rscript -e "remotes::install_github('biobricks-ai/bricktools')"
