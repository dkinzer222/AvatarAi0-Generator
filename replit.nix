{pkgs}: {
  deps = [
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.libGLU
    pkgs.libGL
    pkgs.iana-etc
    pkgs.unzip
    pkgs.postgresql
    pkgs.openssl
  ];
}
