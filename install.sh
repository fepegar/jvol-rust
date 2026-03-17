#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-fepegar/jvol-rust}"
BINARY_NAME="jvol-rust"
INSTALL_DIR="${INSTALL_DIR:-}"
VERSION="${VERSION:-latest}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' is required to install ${BINARY_NAME}." >&2
        exit 1
    fi
}

detect_target() {
    local os
    local arch

    case "$(uname -s)" in
        Linux)
            os="unknown-linux-gnu"
            ;;
        Darwin)
            os="apple-darwin"
            ;;
        *)
            echo "Error: unsupported operating system '$(uname -s)'." >&2
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64 | amd64)
            arch="x86_64"
            ;;
        arm64 | aarch64)
            arch="aarch64"
            ;;
        *)
            echo "Error: unsupported architecture '$(uname -m)'." >&2
            exit 1
            ;;
    esac

    case "${arch}-${os}" in
        x86_64-unknown-linux-gnu | x86_64-apple-darwin | aarch64-apple-darwin)
            printf '%s\n' "${arch}-${os}"
            ;;
        *)
            echo "Error: no pre-built release is available for ${arch}-${os}." >&2
            exit 1
            ;;
    esac
}

resolve_install_dir() {
    if [ -n "${INSTALL_DIR}" ]; then
        printf '%s\n' "${INSTALL_DIR}"
        return
    fi

    if [ -w "/usr/local/bin" ]; then
        printf '%s\n' "/usr/local/bin"
        return
    fi

    printf '%s\n' "${HOME}/.local/bin"
}

download_url_for() {
    local target="$1"

    if [ "${VERSION}" = "latest" ]; then
        printf 'https://github.com/%s/releases/latest/download/%s-%s.tar.gz\n' \
            "${REPO}" "${BINARY_NAME}" "${target}"
        return
    fi

    local tag="${VERSION}"
    case "${tag}" in
        v*)
            ;;
        *)
            tag="v${tag}"
            ;;
    esac

    printf 'https://github.com/%s/releases/download/%s/%s-%s.tar.gz\n' \
        "${REPO}" "${tag}" "${BINARY_NAME}" "${target}"
}

main() {
    require_command curl
    require_command tar
    require_command mktemp

    local target
    local install_dir
    local archive_url
    local tmpdir

    target="$(detect_target)"
    install_dir="$(resolve_install_dir)"
    archive_url="$(download_url_for "${target}")"
    tmpdir="$(mktemp -d)"

    trap 'rm -rf "${tmpdir}"' EXIT

    mkdir -p "${install_dir}"

    if [ ! -w "${install_dir}" ]; then
        echo "Error: install directory '${install_dir}' is not writable." >&2
        echo "Set INSTALL_DIR to a writable path or re-run with elevated privileges." >&2
        exit 1
    fi

    echo "Downloading ${BINARY_NAME} for ${target}..."
    curl -fsSL "${archive_url}" -o "${tmpdir}/${BINARY_NAME}.tar.gz"

    echo "Installing to ${install_dir}..."
    tar -xzf "${tmpdir}/${BINARY_NAME}.tar.gz" -C "${tmpdir}" "${BINARY_NAME}"
    install -m 755 "${tmpdir}/${BINARY_NAME}" "${install_dir}/${BINARY_NAME}"

    echo "Installed ${BINARY_NAME} to ${install_dir}/${BINARY_NAME}"

    case ":${PATH}:" in
        *:"${install_dir}":*)
            ;;
        *)
            echo "Note: ${install_dir} is not currently on your PATH." >&2
            ;;
    esac
}

main "$@"
