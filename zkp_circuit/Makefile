# ShieldPoint ZKP Circuit — Build Automation
# =============================================
# Usage:
#   make setup          — Install Circom + SnarkJS + npm deps
#   make compile        — Compile circuit to R1CS + WASM
#   make trusted-setup  — Run Groth16 trusted setup (8 contributors)
#   make test-vectors   — Generate and verify test vectors
#   make test           — Run Python integration tests
#   make test-all       — Run both integration and tool_registry tests
#   make solidity       — Export Solidity verifier contract
#   make clean          — Remove build artifacts (keeps keys)
#   make clean-all      — Remove everything including keys

SHELL := /bin/bash
PYTHON ?= python3
CIRCUIT_NAME := policy_validity
CROSS_AGENT_CIRCUIT := cross_agent_claim_limit
FRAUD_CIRCUIT := non_membership
CIRCUIT_DIR := circuits
BUILD_DIR := build
KEYS_DIR := keys
TEST_DIR := test_vectors
SCRIPTS_DIR := scripts

# Constraint power for Powers of Tau (2^17 = 131072 max constraints — enough
# for the non-membership circuit at depth=20, which is ~10K constraints)
PTAU_POWER := 17
NUM_CONTRIBUTORS := 8

.PHONY: setup compile trusted-setup test-vectors test test-all solidity clean clean-all info \
        compile-cross cross-trusted-setup test-cross test-cross-all \
        compile-fraud fraud-trusted-setup test-fraud test-fraud-all

info:
        @echo "=== ShieldPoint ZKP Circuit Build System ==="
        @echo "Circuit:       $(CIRCUIT_NAME).circom"
        @echo "Build dir:     $(BUILD_DIR)/"
        @echo "Keys dir:      $(KEYS_DIR)/"
        @echo "Contributors:  $(NUM_CONTRIBUTORS)"
        @echo ""
        @echo "Run 'make compile' to build the circuit."
        @echo "Run 'make trusted-setup' after compile."

# ---- Setup ----
setup:
        @echo "Installing Circom 2.x..."
        @which circom || (curl -L -o /tmp/circom-linux-amd64 https://github.com/iden3/circom/releases/download/v2.1.9/circom-linux-amd64 && \
                chmod +x /tmp/circom-linux-amd64 && mv /tmp/circom-linux-amd64 /usr/local/bin/circom)
        @echo "Installing SnarkJS..."
        npm install -g snarkjs
        @echo "Installing circomlib..."
        npm install circomlib circomlibjs
        @echo "Setup complete."

# ---- Compile ----
compile:
        @echo "Compiling $(CIRCUIT_NAME).circom..."
        circom $(CIRCUIT_DIR)/$(CIRCUIT_NAME).circom --r1cs --wasm --sym -o $(BUILD_DIR)
        @echo ""
        @echo "Circuit info:"
        snarkjs r1cs info $(BUILD_DIR)/$(CIRCUIT_NAME).r1cs

# ---- Trusted Setup ----
trusted-setup: compile
        @echo "=== Phase 1: Powers of Tau ==="
        @mkdir -p $(KEYS_DIR)
        snarkjs powersoftau new bn128 $(PTAU_POWER) $(KEYS_DIR)/pot12_00000.ptau -v
        @for i in $$(seq 1 $(NUM_CONTRIBUTORS)); do \
                echo "Phase 1 contribution $$i/$(NUM_CONTRIBUTORS)..."; \
                snarkjs powersoftau contribute $(KEYS_DIR)/pot12_0000$$((i-1)).ptau \
                        $(KEYS_DIR)/pot12_0000$$i.ptau \
                        --name="ShieldPoint contributor $$i" \
                        -e="random entropy $$i"; \
        done
        snarkjs powersoftau verify $(KEYS_DIR)/pot12_0000$(NUM_CONTRIBUTORS).ptau
        @echo "Preparing Phase 2..."
        snarkjs pt2 $(KEYS_DIR)/pot12_0000$(NUM_CONTRIBUTORS).ptau $(KEYS_DIR)/pot12_final.ptau
        @echo ""
        @echo "=== Phase 2: Circuit-specific setup ==="
        snarkjs g16s $(BUILD_DIR)/$(CIRCUIT_NAME).r1cs $(KEYS_DIR)/pot12_final.ptau $(KEYS_DIR)/circuit_0000.zkey
        @for i in $$(seq 1 $(NUM_CONTRIBUTORS)); do \
                echo "Phase 2 contribution $$i/$(NUM_CONTRIBUTORS)..."; \
                snarkjs zkc $(KEYS_DIR)/circuit_000$$((i-1)).zkey \
                        $(KEYS_DIR)/circuit_000$$i.zkey \
                        -e="v2 contributor $$i"; \
        done
        cp $(KEYS_DIR)/circuit_000$(NUM_CONTRIBUTORS).zkey $(KEYS_DIR)/circuit_final.zkey
        snarkjs zkev $(KEYS_DIR)/circuit_final.zkey $(KEYS_DIR)/verification_key.json
        @echo "Verifying zkey..."
        snarkjs zkv $(BUILD_DIR)/$(CIRCUIT_NAME).r1cs $(KEYS_DIR)/pot12_final.ptau $(KEYS_DIR)/circuit_final.zkey
        @echo "Trusted setup complete!"

# ---- Test Vectors ----
test-vectors:
        @mkdir -p $(TEST_DIR)
        node generate_test_vectors.js

# ---- Tests ----
test:
        $(PYTHON) -m pytest test_integration.py -v

test-all:
        $(PYTHON) -m pytest test_integration.py -v
        $(PYTHON) -m pytest tests/test_zkp_tools.py -v

# ---- Solidity Verifier ----
solidity:
        snarkjs zkesv $(KEYS_DIR)/circuit_final.zkey $(BUILD_DIR)/verifier.sol
        @echo "Solidity verifier exported to $(BUILD_DIR)/verifier.sol"
        @wc -l $(BUILD_DIR)/verifier.sol

# ---- Cross-Agent Circuit (SP-304) ----
# Compiles the cross_agent_claim_limit.circom circuit. Build artifacts
# live in build/cross_agent_claim_limit{.r1cs,.sym,_js/} and trusted-setup
# keys in keys/cross_agent_circuit_*.{zkey,json} so they don't clobber
# the policy_validity artifacts.
compile-cross:
        @echo "Compiling $(CROSS_AGENT_CIRCUIT).circom..."
        circom $(CIRCUIT_DIR)/$(CROSS_AGENT_CIRCUIT).circom --r1cs --wasm --sym -o $(BUILD_DIR)
        @echo ""
        @echo "Circuit info (constraints must be < 30K per SP-304 AC):"
        snarkjs r1cs info $(BUILD_DIR)/$(CROSS_AGENT_CIRCUIT).r1cs

cross-trusted-setup: compile-cross
        @echo "=== Cross-agent circuit: Phase 1 (Powers of Tau, shared) ==="
        @mkdir -p $(KEYS_DIR)
        @if [ ! -f $(KEYS_DIR)/pot12_final.ptau ]; then \
                echo "No shared ptau found - running Phase 1 from scratch..."; \
                snarkjs powersoftau new bn128 $(PTAU_POWER) $(KEYS_DIR)/pot12_00000.ptau -v; \
                for i in $$(seq 1 $(NUM_CONTRIBUTORS)); do \
                        echo "Phase 1 contribution $$i/$(NUM_CONTRIBUTORS)..."; \
                        snarkjs powersoftau contribute $(KEYS_DIR)/pot12_0000$$((i-1)).ptau \
                                $(KEYS_DIR)/pot12_0000$$i.ptau \
                                --name="ShieldPoint contributor $$i" \
                                -e="random entropy $$i"; \
                done; \
                snarkjs powersoftau verify $(KEYS_DIR)/pot12_0000$(NUM_CONTRIBUTORS).ptau; \
                snarkjs pt2 $(KEYS_DIR)/pot12_0000$(NUM_CONTRIBUTORS).ptau $(KEYS_DIR)/pot12_final.ptau; \
        else \
                echo "Reusing shared $(KEYS_DIR)/pot12_final.ptau"; \
        fi
        @echo "=== Cross-agent circuit: Phase 2 (circuit-specific) ==="
        snarkjs g16s $(BUILD_DIR)/$(CROSS_AGENT_CIRCUIT).r1cs $(KEYS_DIR)/pot12_final.ptau $(KEYS_DIR)/cross_agent_circuit_0000.zkey
        @for i in $$(seq 1 $(NUM_CONTRIBUTORS)); do \
                echo "Phase 2 contribution $$i/$(NUM_CONTRIBUTORS)..."; \
                snarkjs zkc $(KEYS_DIR)/cross_agent_circuit_000$$((i-1)).zkey \
                        $(KEYS_DIR)/cross_agent_circuit_000$$i.zkey \
                        -e="cross-agent contributor $$i"; \
        done
        cp $(KEYS_DIR)/cross_agent_circuit_000$(NUM_CONTRIBUTORS).zkey $(KEYS_DIR)/circuit_final.zkey
        snarkjs zkev $(KEYS_DIR)/circuit_final.zkey $(KEYS_DIR)/verification_key.json
        @echo "Cross-agent trusted setup complete!"

test-cross:
        $(PYTHON) -m pytest test_cross_agent.py -v

test-cross-all: test-cross
        $(PYTHON) -m pytest test_integration.py -v
        $(PYTHON) -m pytest ../shieldpoint_agents/tests/test_cross_agent_zkp.py -v

# ---- Fraud Detection Circuit (SP-500 / SP-501) ----
# Compiles the non_membership.circom circuit for cross-party fraud detection.
# Build artifacts in build/non_membership{.r1cs,.sym,_js/} and trusted-setup
# keys in keys/non_membership_*.{zkey,json}.
compile-fraud:
        @echo "Compiling $(FRAUD_CIRCUIT).circom..."
        @mkdir -p $(BUILD_DIR)/non_membership
        circom $(CIRCUIT_DIR)/$(FRAUD_CIRCUIT).circom --r1cs --wasm --sym -o $(BUILD_DIR)/non_membership
        @echo ""
        @echo "Circuit info (constraints must be < 80K per SP-501 AC):"
        snarkjs r1cs info $(BUILD_DIR)/non_membership/$(FRAUD_CIRCUIT).r1cs

fraud-trusted-setup: compile-fraud
        @echo "=== Fraud circuit: Phase 1 (Powers of Tau, shared) ==="
        @mkdir -p $(KEYS_DIR)
        @if [ ! -f $(KEYS_DIR)/pot12_final.ptau ]; then \
                echo "No shared ptau found - running Phase 1 from scratch..."; \
                snarkjs powersoftau new bn128 $(PTAU_POWER) $(KEYS_DIR)/pot12_00000.ptau -v; \
                for i in $$(seq 1 $(NUM_CONTRIBUTORS)); do \
                        echo "Phase 1 contribution $$i/$(NUM_CONTRIBUTORS)..."; \
                        snarkjs powersoftau contribute $(KEYS_DIR)/pot12_0000$$((i-1)).ptau \
                                $(KEYS_DIR)/pot12_0000$$i.ptau \
                                --name="ShieldPoint contributor $$i" \
                                -e="random entropy $$i"; \
                done; \
                snarkjs powersoftau verify $(KEYS_DIR)/pot12_0000$(NUM_CONTRIBUTORS).ptau; \
                snarkjs pt2 $(KEYS_DIR)/pot12_0000$(NUM_CONTRIBUTORS).ptau $(KEYS_DIR)/pot12_final.ptau; \
        else \
                echo "Reusing shared $(KEYS_DIR)/pot12_final.ptau"; \
        fi
        @echo "=== Fraud circuit: Phase 2 (circuit-specific) ==="
        snarkjs g16s $(BUILD_DIR)/non_membership/$(FRAUD_CIRCUIT).r1cs $(KEYS_DIR)/pot12_final.ptau $(KEYS_DIR)/non_membership_0000.zkey
        @for i in $$(seq 1 $(NUM_CONTRIBUTORS)); do \
                echo "Phase 2 contribution $$i/$(NUM_CONTRIBUTORS)..."; \
                snarkjs zkc $(KEYS_DIR)/non_membership_000$$((i-1)).zkey \
                        $(KEYS_DIR)/non_membership_000$$i.zkey \
                        -e="fraud contributor $$i"; \
        done
        cp $(KEYS_DIR)/non_membership_000$(NUM_CONTRIBUTORS).zkey $(KEYS_DIR)/non_membership_final.zkey
        snarkjs zkev $(KEYS_DIR)/non_membership_final.zkey $(KEYS_DIR)/non_membership_verification_key.json
        @echo "Fraud detection trusted setup complete!"

test-fraud:
        $(PYTHON) -m pytest fraud_detection/tests/ -v

test-fraud-all: test-fraud
        $(PYTHON) -m pytest ../tests/v2/test_fraud_detection.py -v

# ---- Clean ----
clean:
        rm -rf $(BUILD_DIR)/$(CIRCUIT_NAME).r1cs
        rm -rf $(BUILD_DIR)/$(CIRCUIT_NAME).sym
        rm -rf $(BUILD_DIR)/$(CIRCUIT_NAME)_js/
        rm -rf $(BUILD_DIR)/$(CROSS_AGENT_CIRCUIT).r1cs
        rm -rf $(BUILD_DIR)/$(CROSS_AGENT_CIRCUIT).sym
        rm -rf $(BUILD_DIR)/$(CROSS_AGENT_CIRCUIT)_js/
        rm -rf $(BUILD_DIR)/non_membership/
        rm -rf $(TEST_DIR)/
        rm -f $(BUILD_DIR)/verifier.sol

clean-all: clean
        rm -f $(KEYS_DIR)/circuit_*.zkey
        rm -f $(KEYS_DIR)/cross_agent_circuit_*.zkey
        rm -f $(KEYS_DIR)/non_membership_*.zkey
        rm -f $(KEYS_DIR)/non_membership_verification_key.json
        rm -f $(KEYS_DIR)/pot12_*.ptau
        rm -f $(KEYS_DIR)/verification_key.json
