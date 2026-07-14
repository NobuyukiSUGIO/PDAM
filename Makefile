.PHONY: help test scenarios demo batch ablate clean

help:
	@echo "PDAM testbed — targets:"
	@echo "  make test       run the unit + integration test suite"
	@echo "  make scenarios  generate the 96-scenario Appendix-A matrix"
	@echo "  make demo       run one representative scenario with a trace"
	@echo "  make batch      run all scenarios × defenses, write results/"
	@echo "  make ablate     ablation over difficulty / memory / workload"
	@echo "  make clean      remove generated scenarios, results, caches"

test:
	python3 -m unittest discover -s tests

scenarios:
	python3 -m pdam gen-scenarios scenarios/

demo: scenarios
	python3 -m pdam run scenarios/personal_secretary_A3_hard.json \
		--defense minimal_defense --trace

batch: scenarios
	python3 -m pdam batch --outdir results

ablate:
	python3 -m pdam ablate

clean:
	rm -rf results/*.csv results/*.json scenarios/*.json
	find . -name __pycache__ -type d -exec rm -rf {} +
