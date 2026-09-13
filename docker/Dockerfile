# DSV4-DSpark on Ascend — LingShu-compliant image. ONE Dockerfile, two flavours via ARG.
#
#   ROLE=train  -> conda env dspark-dsv4-compile (torch 2.12.0 / torch_npu 2.12.0.rc1), no vLLM
#   ROLE=serve  -> conda env dspark-dsv4-austin  (torch 2.10.0 / torch_npu 2.10.0) + vLLM 0.23.0
#                  + vllm-ascend-v4, both EDITABLE
#
# ⚠️ EDITABLE INSTALLS BIND TO ABSOLUTE PATHS. A `pip install -e` writes the source dir into a
# .pth file, so the image must recreate those paths EXACTLY or the module vanishes. That is why
# the conda prefix and the vLLM trees are copied to the same absolute paths they had on the box,
# odd-looking usernames and all. Do not "tidy" them into /opt.
#
# ⚠️ THE DRIVER IS NOT IN HERE, BY DESIGN. /usr/local/Ascend/driver is kernel-coupled and comes
# from the HOST at run time (-v ... :ro plus the /dev/davinci* devices). Baking it would break on
# any other host. The CANN TOOLKIT is in the image; the driver is not.
#
# Build with build_image.sh — it stages the copied trees into ./stage first, because a 20 GB
# build context sent from a working tree would be unusable (and .dockerignore excludes everything
# but ./stage for exactly that reason).

ARG BASE_IMAGE=openeuler/openeuler:22.03-lts
FROM ${BASE_IMAGE}

ARG ROLE=train
ARG CONDA_PREFIX_PATH=/home/n84449292/miniconda3
ARG CANN_HOME=/home/a00652497/CANN/9.0.0.0430
ARG SPECULATORS_SHA=c8638e5e
ENV ROLE=${ROLE} \
    TZ=Asia/Shanghai \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# ── 1. LingShu mandatory packages ──────────────────────────────────────────────────────
# openEuler source per https://mirrors.tools.huawei.com/home . `hostname` ships in the
# `hostname` package on openEuler, `ping` in `iputils`, `chpasswd` in `shadow`.
RUN sed -i 's|http://repo.openeuler.org|http://mirrors.tools.huawei.com/openeuler|g' \
        /etc/yum.repos.d/*.repo 2>/dev/null || true; \
    yum makecache && yum install -y \
        openssh-server openssh-clients rsync dos2unix hostname iputils shadow \
        bash tar gzip which findutils procps-ng git ca-certificates \
    && yum clean all && rm -rf /var/cache/yum

# ── 2. LingShu sshd requirements ───────────────────────────────────────────────────────
# PermitRootLogin yes is mandatory; host keys are pre-generated so a minimal image can start
# sshd directly (entrypoint.sh regenerates them if this layer is ever stripped).
RUN mkdir -p /var/run/sshd && \
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    ssh-keygen -A

# ── 3. CANN toolkit (NOT the driver) ───────────────────────────────────────────────────
COPY stage/cann ${CANN_HOME}

# ── 4. conda env, at its ORIGINAL absolute path ────────────────────────────────────────
COPY stage/conda_env ${CONDA_PREFIX_PATH}/envs/${ROLE}_env

# ── 5. editable source trees, also at their ORIGINAL paths (serve only) ────────────────
COPY stage/src /stage_src
RUN if [ -d /stage_src/vllm ]; then \
        mkdir -p /home/a00652497/dspark_austin/installation && \
        mv /stage_src/vllm       /home/a00652497/dspark_austin/installation/vllm-v0.23.0 && \
        mv /stage_src/vllm_ascend /home/a00652497/dspark_austin/installation/vllm-ascend-v4 ; \
    fi; rm -rf /stage_src

# ── 6. speculators: a CLEAN clone, never the 114 GB working tree ───────────────────────
RUN git clone https://github.com/Sawyer117/speculators.git /opt/speculators && \
    cd /opt/speculators && git checkout ${SPECULATORS_SHA} && \
    mkdir -p /home/a00652497/dspark_austin && \
    ln -s /opt/speculators /home/a00652497/dspark_austin/speculators

# ── 7. environment wiring ──────────────────────────────────────────────────────────────
# ★ $CONDA_PREFIX/lib MUST precede the system libs: conda-forge's libsqlite pulls libicui18n,
#   which needs a newer CXXABI than the system libstdc++. Without this, importing torch_npu
#   fails with "Failed to load the backend extension". Costs nothing; saves an hour.
COPY setup_proxy.sh /usr/local/bin/setup_proxy.sh
COPY entrypoint.sh  /usr/local/bin/entrypoint.sh
RUN dos2unix /usr/local/bin/setup_proxy.sh /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/setup_proxy.sh /usr/local/bin/entrypoint.sh && \
    { \
      echo "export PATH=${CONDA_PREFIX_PATH}/envs/${ROLE}_env/bin:\$PATH"; \
      echo "export CONDA_PREFIX=${CONDA_PREFIX_PATH}/envs/${ROLE}_env"; \
      echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"; \
      echo "source ${CANN_HOME}/ascend-toolkit/set_env.sh 2>/dev/null || true"; \
      echo "source /usr/local/bin/setup_proxy.sh"; \
    } >> /root/.bashrc

ENV PATH=${CONDA_PREFIX_PATH}/envs/${ROLE}_env/bin:$PATH \
    CONDA_PREFIX=${CONDA_PREFIX_PATH}/envs/${ROLE}_env \
    LD_LIBRARY_PATH=${CONDA_PREFIX_PATH}/envs/${ROLE}_env/lib \
    ASCEND_TOOLKIT_HOME=${CANN_HOME}/ascend-toolkit/latest

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# ⚠️ `sleep infinity`, NOT `/bin/bash`. The platform starts the container detached and execs in
# afterwards, so there is no TTY -- bash would read EOF on stdin and exit IMMEDIATELY, and the
# container would die. The official check only greps the CMD string for "/bin/bash|sleep", so a
# bash CMD PASSES the check and still fails in practice. The migration guide says the same:
# "sleep infinity is recommended because it reliably keeps non-interactive containers running".
CMD ["sleep", "infinity"]
