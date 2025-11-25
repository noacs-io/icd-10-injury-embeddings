{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = with pkgs; [

    # R requirements
    curl
    libxml2
    pkg-config
    fontconfig
    openssl
    harfbuzz
    freetype
    fribidi
    libpng
    libtiff
    libjpeg
    libwebp
  ];

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.openssl
  ];

  languages.r.enable = true;

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
      };
    };
  };
}
