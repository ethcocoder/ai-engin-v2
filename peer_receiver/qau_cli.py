import sys
import argparse
import numpy as np
from qau_qvs.core.qvs import QVS
from qau_qvs.fields.quantum_fields import QuantumAlgorithms, QuantumCryptography

def main():
    parser = argparse.ArgumentParser(description="Quantum Absolute Unit (QAU) CLI")
    subparsers = parser.add_subparsers(dest="command", help="QVS Instruction")

    # 1. SHOR Command
    shor_parser = subparsers.add_parser("shor", help="Run Shor's Period-Finding Pattern")
    shor_parser.add_argument("--bits", type=int, default=3, help="Number of qubits")

    # 2. GROVER Command
    grover_parser = subparsers.add_parser("grover", help="Run Grover's Search Pattern")
    grover_parser.add_argument("--target", type=str, required=True, help="Target bitstring (e.g., 101)")
    grover_parser.add_argument("--iter", type=int, default=2, help="Number of iterations")

    # 3. E91 Command
    e91_parser = subparsers.add_parser("e91", help="Run E91 Key Exchange (NCB Bond)")

    args = parser.parse_args()
    qvs = QVS()

    if args.command == "shor":
        alg = QuantumAlgorithms(qvs)
        res = alg.shor_factorization_pattern(args.bits)
        print(f"[QAU] Shor period result: {''.join(map(str, res))}")

    elif args.command == "grover":
        alg = QuantumAlgorithms(qvs)
        target = tuple(map(int, args.target))
        res = alg.grover_search_pattern(target, args.iter)
        print(f"[QAU] Grover search result: {''.join(map(str, res))} {'(SUCCESS)' if res == target else '(FAIL)'}")

    elif args.command == "e91":
        crypto = QuantumCryptography(qvs)
        alice, bob = crypto.e91_key_exchange()
        print(f"[QAU] E91 correlation bond formed.")
        print(f"      Alice's Key: {''.join(map(str, alice))}")
        print(f"      Bob's Key:   {''.join(map(str, bob))}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
