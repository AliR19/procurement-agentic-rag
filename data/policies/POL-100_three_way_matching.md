# POL-100 Three-Way Matching Policy

*Fictional policy for a portfolio demo. Not based on any real organization.*

## 1. Purpose
Every supplier invoice must be matched against the purchase order (PO) and the goods receipt before payment is released. This is called three-way matching.

## 2. Matching tolerances
An invoice passes automatic matching only when all of the following hold:
- **Unit price:** the invoiced unit price is within 2% of the PO unit price.
- **Quantity:** the invoiced quantity is within 5% of the received quantity. Invoicing for more than was received is never allowed without approval.
- **Invoice total:** the invoice total differs from the PO total by no more than CAD 50.

## 3. Mismatches
Any line outside tolerance is a discrepancy and the invoice is blocked for payment. A blocked invoice must be documented in an exception report (see POL-300) before it can be released.

## 4. Price increases
A supplier may not bill above the PO unit price unless a PO amendment was approved before delivery. Price increases without an approved amendment are treated as price discrepancies, regardless of the reason given by the supplier.

## 5. Short deliveries
If the invoiced quantity is lower than the ordered quantity, the PO remains partially open. The buyer must confirm with the supplier whether the balance will ship or the PO line should be closed.
