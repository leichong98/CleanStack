# -*- Python -*-

import os

# Setup config name.
config.name = 'CleanStack'

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.c', '.cpp', '.m', '.mm', '.ll', '.test']

# Add clang substitutions.
config.substitutions.append( ("%clang_nocleanstack ", config.clang + " -O0 -fno-sanitize=clean-stack ") )
config.substitutions.append( ("%clang_cleanstack ", config.clang + " -O0 -fsanitize=clean-stack ") )

if config.lto_supported:
  config.substitutions.append((r"%clang_lto_cleanstack ", ' '.join(config.lto_launch + [config.clang] + config.lto_flags + ['-fsanitize=clean-stack '])))

if config.host_os not in ['Linux', 'FreeBSD', 'NetBSD']:
   config.unsupported = True