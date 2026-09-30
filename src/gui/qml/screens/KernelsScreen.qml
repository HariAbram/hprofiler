import QtQuick
import Hprofiler 1.0
import "../components"

// GUI equivalent of the TUI's HotspotsWidget -- a sortable/filterable
// function table built on the shared DataTable component (real Qt
// model/view: Kernels.table is a TableBundle backed by a
// QAbstractListModel + QSortFilterProxyModel, see src/gui/tablemodel.py),
// not the hand-rolled JS sort/filter this screen used before this round.
DataTable {
    id: root
    anchors.fill: parent
    title: "Kernels"
    table: Kernels.table
    isRowSelected: function(row) {
        return Nav.selectedCategory === row.category && Nav.selectedName === row.rawName
    }
    onRowClicked: (row) => Nav.selectFunction(row.category, row.rawName)
}
