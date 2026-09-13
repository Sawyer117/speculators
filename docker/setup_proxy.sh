#!/bin/bash
# Pick the proxy by which subnet the container landed in. Sourced from /root/.bashrc.
# ⚠️ Fill in your own account before use; do NOT commit real credentials.
get_container_ip() {
    local ip=""
    command -v hostname >/dev/null && ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    [ -z "$ip" ] && command -v ip >/dev/null && \
        ip=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | head -n1)
    echo "$ip"
}
container_ip=$(get_container_ip)
if   [[ "$container_ip" == 80.* ]]; then export PROXY_HOST=80.5.1.30
elif [[ "$container_ip" == 90.* ]]; then export PROXY_HOST=90.91.191.181
else                                     export PROXY_HOST=141.61.33.201
fi
# ⚠️ 账号密码从环境注入,不写死在镜像里:docker run -e PROXY_USER=... -e PROXY_PASS=...
if [ -n "${PROXY_USER:-}" ] && [ -n "${PROXY_PASS:-}" ]; then
    export http_proxy="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:8091"
    export https_proxy="$http_proxy"
fi
export no_proxy=127.0.0.1,localhost,local,.local,.huawei.com
export NO_PROXY="$no_proxy"
export GIT_SSL_NO_VERIFY=1
