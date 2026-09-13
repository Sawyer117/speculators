# DSV4-DSpark on Ascend — LingShu-compliant image. One Dockerfile, ROLE picks the flavour.
#
# ⚠️ FOUR THINGS THIS FILE ENCODES THAT ARE NOT OBVIOUS. Each cost real time to find.
#
# 1. BASE_IMAGE MUST ALREADY BE LOCAL. Neither build box can pull: one has no DNS at all
#    (resolver at [::1]:53 refuses), the other times out to the internal harbor. `FROM` runs
#    before any proxy you pass can help, so the base has to come from `docker images`.
#
# 2. THE BASE CARRIES A DEAD PROXY IN ITS ENV. torchspec_ty:116 ships
#    http_proxy=http://90.255.89.236:3128, which is unreachable from these hosts, so apt hangs
#    on it. It is in the ENVIRONMENT, not apt.conf.d -- passing --build-arg http_proxy
#    overrides it (the RUN below also clears any apt.conf.d proxy snippet for good measure).
#
# 3. EDITABLE INSTALLS BIND TO ABSOLUTE PATHS. `pip install -e` writes the source dir into a
#    .pth file, so the image must recreate those paths EXACTLY -- odd usernames included.
#    Do not "tidy" them into /opt; the module simply disappears if you do.
#
# 4. THE DRIVER IS DELIBERATELY ABSENT. /usr/local/Ascend/driver is kernel-coupled and is
#    bind-mounted from the host at run time together with /dev/davinci*. Baking it in breaks
#    on every other host. The CANN TOOLKIT is in the image; the driver is not.
#
# Build through build_image.sh, which stages the copied trees with hard links first -- docker
# 18.09 ships the whole context to the daemon, and the trees live under three different homes.

ARG BASE_IMAGE=torchspec_ty:116
FROM ${BASE_IMAGE}

ARG ROLE=serve
ARG CONDA_ROOT=/home/n84449292/miniconda3
ARG CONDA_ENV=dspark-dsv4-serving
ARG CANN_HOME=/home/a00652497/CANN/9.1.0.0627
ARG SKIP_PKGS=0

ENV TZ=Asia/Shanghai LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive

# ── 1. LingShu mandatory packages ──────────────────────────────────────────────────────
# Ubuntu base -> apt. Set SKIP_PKGS=1 for a base that already has all six (ascend-verl:v4),
# which then needs no network at all.
RUN if [ "$SKIP_PKGS" = "1" ]; then echo ">>> SKIP_PKGS=1, 跳过装包"; else \
      rm -f /etc/apt/apt.conf.d/*proxy* 2>/dev/null; \
      ( command -v apt-get >/dev/null \
        && apt-get update -qq \
        && apt-get install -y -qq --no-install-recommends \
             openssh-server rsync dos2unix hostname iputils-ping passwd \
        && rm -rf /var/lib/apt/lists/* ) \
      || ( ( command -v dnf >/dev/null && M=dnf ) || M=yum; \
           $M install -y openssh-server rsync dos2unix hostname iputils shadow && $M clean all ); \
    fi

# ── 2. LingShu sshd requirements ───────────────────────────────────────────────────────
RUN mkdir -p /var/run/sshd && \
    sed -ri 's/^[#[:space:]]*PermitRootLogin[[:space:]].*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    grep -q '^PermitRootLogin yes' /etc/ssh/sshd_config || echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    ssh-keygen -A && /usr/sbin/sshd -t

# ── 3. CANN toolkit (NOT the driver) ───────────────────────────────────────────────────
COPY cann ${CANN_HOME}

# ── 4. conda env, at its ORIGINAL absolute path ────────────────────────────────────────
COPY conda_env ${CONDA_ROOT}/envs/${CONDA_ENV}

# ── 5. editable source trees, also at their ORIGINAL paths ─────────────────────────────
# build_image.sh stages each tree under stage/src/<basename-of-its-parent>__<basename>, and
# writes stage/src/PATHS mapping staged dir -> final absolute path, so this layer is generic.
COPY src /stage_src
RUN if [ -f /stage_src/PATHS ]; then \
      while IFS='|' read -r d dest; do \
        [ -z "$d" ] && continue; \
        mkdir -p "$(dirname "$dest")" && mv "/stage_src/$d" "$dest" && echo "  -> $dest"; \
      done < /stage_src/PATHS ; \
    fi; \
    cp -f /stage_src/GIT_MANIFEST /opt/image_manifest.txt 2>/dev/null || true; \
    rm -rf /stage_src

# ── 6. environment wiring ──────────────────────────────────────────────────────────────
# ★ $CONDA_PREFIX/lib MUST precede the system libs. conda-forge's libsqlite pulls libicui18n,
#   which wants a newer CXXABI than the system libstdc++; without this, `import torch_npu`
#   dies with "Failed to load the backend extension". One line, one hour saved.
# ★ The base image's dead proxy is cleared so it cannot poison anything at run time.
COPY setup_proxy.sh /usr/local/bin/setup_proxy.sh
COPY entrypoint.sh  /usr/local/bin/entrypoint.sh
RUN dos2unix /usr/local/bin/setup_proxy.sh /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/setup_proxy.sh /usr/local/bin/entrypoint.sh && \
    { \
      echo "export PATH=${CONDA_ROOT}/envs/${CONDA_ENV}/bin:\$PATH"; \
      echo "export CONDA_PREFIX=${CONDA_ROOT}/envs/${CONDA_ENV}"; \
      echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"; \
      echo "source ${CANN_HOME}/ascend-toolkit/set_env.sh 2>/dev/null || true"; \
      echo "source /usr/local/bin/setup_proxy.sh"; \
    } >> /root/.bashrc

ENV http_proxy="" https_proxy="" \
    PATH=${CONDA_ROOT}/envs/${CONDA_ENV}/bin:$PATH \
    CONDA_PREFIX=${CONDA_ROOT}/envs/${CONDA_ENV} \
    LD_LIBRARY_PATH=${CONDA_ROOT}/envs/${CONDA_ENV}/lib \
    CANN_HOME=${CANN_HOME} \
    ROLE=${ROLE}

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# ⚠️ `sleep infinity`, NOT `/bin/bash`. The platform starts the container detached and execs in
# afterwards, so there is no TTY -- bash would read EOF on stdin and exit IMMEDIATELY, killing
# the container. The official check only greps the CMD string for "/bin/bash|sleep", so a bash
# CMD PASSES the check and still fails in practice.
CMD ["sleep", "infinity"]
