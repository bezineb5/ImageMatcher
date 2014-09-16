var imageMatcherApp = angular.module('imageMatcherApp', []);

imageMatcherApp.controller('ImageListController', function ($scope, $http) {
  $http.get('references/').success(function(data) {
    $scope.referenceImages = data;
  });
});
