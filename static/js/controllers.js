var phonecatApp = angular.module('imageMatcherApp', []);

phonecatApp.controller('ImageListController', function ($scope, $http) {
  $http.get('references/').success(function(data) {
    $scope.referenceImages = data;
  });
});
