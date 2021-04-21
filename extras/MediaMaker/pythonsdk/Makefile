# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2021 Blue BrainProject / EPFL
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

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
MIN_COV?=80
VENV_INSTALLED=.installed
PIP_INSTALL_OPTIONS=--ignore-installed

FIND_LINT_PY=`find mediamaker -name "*.py" -not -path "*/*test*"`
LINT_PYFILES := $(shell find $(FIND_LINT_PY))

$(VENV):
	virtualenv --system-site-packages $(VENV)

$(VENV_INSTALLED): $(VENV)
	# Brayns 1.0.0 limitation, use pip 9.0.2
	$(VENV_BIN)/pip install pip==9.0.2
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
	$(VENV_BIN)/nosetests -v --with-coverage --cover-min-percentage=$(MIN_COV) --cover-erase --cover-package mediamaker

run_tests_xunit: $(VENV_INSTALLED)
	@mkdir -p $(ROOT_DIR)/test-reports
	$(VENV_BIN)/nosetests mediamaker --with-coverage --cover-min-percentage=$(MIN_COV) --cover-inclusive --cover-erase --cover-package=mediamaker --with-xunit --xunit-file=test-reports/nosetests_mediamaker.xml

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
	@rm -rf mediamaker/mediamaker.egg-info
	@rm -f .coverage
	@rm -rf test-reports
	@rm -rf dist
	@rm -f $(VENV_INSTALLED)

.PHONY: run_pycodestyle test clean_test_venv clean doc
