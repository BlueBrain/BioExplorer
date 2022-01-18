# Copyright (c) 2020-2021, Blue Brain Project
#                          Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of BioExplorer <https://github.com/BlueBrain/BioExplorer>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# All rights reserved. Do not distribute without further notice.

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#see if we're in a virtualenv, and use that, otherwise use the default
ifdef VIRTUAL_ENV
   VENV=$(VIRTUAL_ENV)
else
   VENV:=venv
endif
VENV_BIN:=$(VENV)/bin

# simulate running in headless mode
unexport DISPLAY

# Test coverage pass threshold (percent)
MIN_COV?=60
VENV_INSTALLED=.installed
PIP_INSTALL_OPTIONS=--ignore-installed

FIND_LINT_PY=`find bioexplorer -name "*.py" -not -path "*/*test*"`
LINT_PYFILES := $(shell find $(FIND_LINT_PY))

$(VENV):
	virtualenv --system-site-packages $(VENV)

$(VENV_INSTALLED): $(VENV)
	# Brayns 1.0.0 limitation, use pip 9.0.2
	$(VENV_BIN)/pip install pip==9.0.2
	$(VENV_BIN)/pip install brayns==1.0.0
	$(VENV_BIN)/pip install pip --upgrade pip
	$(VENV_BIN)/pip install $(PIP_INSTALL_OPTIONS) -r requirements_dev.txt
	$(VENV_BIN)/pip install -e .
	touch $@

run_pycodestyle: $(VENV_INSTALLED)
	$(VENV_BIN)/pycodestyle $(LINT_PYFILES) > pycodestyle.txt

run_pydocstyle: $(VENV_INSTALLED)
	$(VENV_BIN)/pydocstyle $(LINT_PYFILES) > pydocstyle.txt

run_pylint: $(VENV_INSTALLED)
	$(VENV_BIN)/pylint --rcfile=pylintrc $(LINT_PYFILES) > pylint.txt

run_tests: $(VENV_INSTALLED)
	$(VENV_BIN)/nosetests -v --with-coverage --cover-min-percentage=$(MIN_COV) --cover-erase --cover-package bioexplorer

run_tests_xunit: $(VENV_INSTALLED)
	@mkdir -p $(ROOT_DIR)/test-reports
	$(VENV_BIN)/nosetests bioexplorer --with-coverage --cover-min-percentage=$(MIN_COV) --cover-inclusive --cover-erase --cover-package=bioexplorer --with-xunit --xunit-file=test-reports/nosetests_bioexplorer.xml

lint: run_pycodestyle run_pydocstyle run_pylint

test: lint run_tests

doc: $(VENV_INSTALLED)
	make SPHINXBUILD=$(VENV_BIN)/sphinx-build -C doc html

doc_pdf: $(VENV_INSTALLED)
	make SPHINXBUILD=$(VENV_BIN)/sphinx-build -C doc latexpdf

clean_test_venv:
	@rm -rf $(VENV_INSTALLED)
	@rm -rf $(ROOT_DIR)/test-reports

clean_doc:
	@test -x $(VENV_BIN)/sphinx-build && make SPHINXBUILD=$(VENV_BIN)/sphinx-build  -C doc clean || true
	@rm -rf $(ROOT_DIR)/doc/build

clean: clean_doc clean_test_venv
	@rm -f pycodestyle.txt
	@rm -f pydocstyle.txt
	@rm -f pylint.txt
	@rm -rf bioexplorer/bioexplorer.egg-info
	@rm -f .coverage
	@rm -rf test-reports
	@rm -rf dist
	@rm -f $(VENV_INSTALLED)

.PHONY: run_pycodestyle test clean_test_venv clean doc
